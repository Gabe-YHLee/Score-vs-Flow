import torch
import copy

class ScoreBasedModels(torch.nn.Module):
    def __init__(
        self, 
        score_model: torch.nn.Module,
        x_dim: int,
        beta_max: float = 1.0,
        ):
        super(ScoreBasedModels, self).__init__()
        # Variance preserving diffusion path is implemented 
        # Forward pass: dx = -0.5 beta_t x dt + sqrt(beta_t) dw_t
        # We assume beta_t = beta_max(t)
        # P_t(x|x_0) = N(x; x_0e^{-1/4 beta_max t^2}, (1 - e^{-1/2 beta_max t^2})I)
        # grad_x logP_t(x|x_0) = -(x-x_0e^{-1/4 beta_max t^2})/(1 - e^{-1/2 beta_max t^2})
        
        self.score_model = score_model
        self.x_dim = x_dim
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