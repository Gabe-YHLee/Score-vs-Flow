import torch
import copy

class ScoreBasedModels(torch.nn.Module):
    def __init__(
        self, 
        score_model: torch.nn.Module,
        beta_max: float = 1.0,
        ):
        super(ScoreBasedModels, self).__init__()
        # Variance preserving diffusion path is implemented 
        # Forward pass: dx = -0.5 beta_t x dt + sqrt(beta_t) dw_t
        # We assume beta_t = beta_max(t)
        # P_t(x|x_0) = N(x; x_0e^{-1/4 beta_max t^2}, (1 - e^{-1/2 beta_max t^2})I)
        # grad_x logP_t(x|x_0) = -(x-x_0e^{-1/4 beta_max t^2})/(1 - e^{-1/2 beta_max t^2})
        
        self.score_model = score_model
        self.beta_max = beta_max
    
    def beta(self, t):
        return self.beta_max * t
    
    def sample_from_p_tx_x0(self, t, x0):
        mu = x0 * torch.exp(-0.25 * self.beta_max * t**2)
        std = torch.sqrt(1 - torch.exp(-0.5 * self.beta_max * t**2))
        return torch.randn_like(x0) * std + mu
    
    def score_conditioned_on_x1(self, t, x, x0):
        t = t.clamp(min=1.0e-3, max=1.0e10)
        mu = x0 * torch.exp(-0.25 * self.beta_max * t**2)
        std = torch.sqrt(1 - torch.exp(-0.5 * self.beta_max * t**2))
        return -(x - mu) / std**2
    
    def velocity_conditioned_on_x1(self, t, x, x0):
        # −10𝑥𝑡−10𝑡𝑠_𝜃 (𝑡,𝑥)
        beta_t = self.beta(t)
        score_t = self.score_conditioned_on_x1(t, x, x0)
        return 0.5*beta_t*x + 0.5*beta_t*score_t
    
    def corruption_process(self, x0, t):
        xt = self.sample_from_p_tx_x0(t, x0)
        return xt
    
    def train_step(self, x0, optimizer, *args, **kwargs):
        optimizer.zero_grad()
        
        t = torch.rand(len(x0)).view(-1, 1).clamp(min=1e-3, max=1e10).to(x0)
        xt = self.sample_from_p_tx_x0(t, x0)
        score_label = self.score_conditioned_on_x1(t, xt, x0)
        
        loss = ((self.score_model(torch.cat([t, xt], dim=-1)) - score_label)**2).mean()
        
        loss.backward()
        optimizer.step()
        return {"loss": loss.detach().cpu().item()}
    
    def backward_process(self, x1, dt=0.01, mode='sde'):
        T = int(1/dt)
        xt = copy.deepcopy(x1)
        xtraj = [copy.deepcopy(xt).unsqueeze(1)]
        for t in torch.linspace(1, 0, T):
            t = t.clamp(min=1e-3, max=1e10).to(x1).view(-1, 1).repeat(len(x1), 1)
            beta_t = self.beta(t)
            score_t =  self.score_model(torch.cat([t, xt], dim=-1))
            if mode == 'sde':
                xt += (0.5*beta_t*xt + beta_t*score_t)*dt + torch.sqrt(beta_t * dt) * torch.randn_like(xt)
            elif mode == 'ode':
                xt += (0.5*beta_t*xt + 0.5 * beta_t*score_t)*dt
            xtraj.append(copy.deepcopy(xt.detach()).unsqueeze(1))
        return xt, torch.cat(xtraj, dim=1)
    
class ScoreBasedModelsV2(torch.nn.Module):
    def __init__(
        self, 
        score_model: torch.nn.Module,
        mode: dict = {
            "type": "ve",
            "sigma_min": 0.01,
            "sigma_max": 1.0,
            "schedule": "linear",
            },
        ):
        super(ScoreBasedModelsV2, self).__init__()
        # VE, VP, and sub-VP
        # various noise schedule implemented
        self.score_model = score_model
        self.mode = mode
    
    def linear_schedule(self, t, a_min, a_max):
        return a_min + t * (a_max - a_min)
    
    def sigmoid_schedule(self, t, a_min, a_max, T=10):
        return a_min + (a_max - a_min) * torch.sigmoid((t - 0.5)/T)
    
    def cosine_schedule(self, t, a_min, a_max):
        return a_max - 0.5 * (a_max - a_min) * (1 + torch.cos(
            t * 3.141592653589793))
    
    def quadratic_schedule(self, t, a_min, a_max):
        return a_min + (a_max - a_min) * (t**2)
    
    def sample_from_p_tx_x0(self, t, x0):
        if self.mode["type"] == "ve":
            mu_t = x0
            sigma_1 = self.mode["sigma_max"]
            sigma_0 = self.mode["sigma_min"]
            if self.mode["schedule"] == "linear":
                sigma_t = self.linear_schedule(t, sigma_0, sigma_1)
            elif self.mode["schedule"] == "sigmoid":
                T = self.mode["T"]
                sigma_t = self.sigmoid_schedule(t, sigma_0, sigma_1, T=T)
            elif self.mode["schedule"] == "cosine":
                sigma_t = self.cosine_schedule(t, sigma_0, sigma_1)
            elif self.mode["schedule"] == "quadratic":
                sigma_t = self.quadratic_schedule(t, sigma_0, sigma_1)
            std_t = torch.sqrt((sigma_t**2 - sigma_0**2).clamp(min=0, max=1e10))
        elif self.mode["type"] == "vp":
            # let b(t) = int_0^t beta(s) ds
            b_max = self.mode["b_max"]
            if self.mode["schedule"] == "linear":
                b = self.linear_schedule(t, 0, b_max)
            elif self.mode["schedule"] == "sigmoid":
                T = self.mode["T"]
                b = self.sigmoid_schedule(t, 0, b_max, T=T)
            elif self.mode["schedule"] == "cosine":
                b = self.cosine_schedule(t, 0, b_max)
            elif self.mode["schedule"] == "quadratic":
                b = self.quadratic_schedule(t, 0, b_max)
            mu_t = x0 * torch.exp(-0.5 * b)
            std_t = torch.sqrt((1 - torch.exp(-b)).clamp(min=0, max=1e10))
        return torch.randn_like(x0) * std_t + mu_t
    
    def corruption_process(self, x0, t):
        xt = self.sample_from_p_tx_x0(t, x0)
        return xt