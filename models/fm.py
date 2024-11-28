import torch
import copy
class FlowBasedModels(torch.nn.Module):
    def __init__(
        self,
        velocity_model: torch.nn.Module,
        sigma_min: float = 0.01,
    ):
        super(FlowBasedModels, self).__init__()
        # Flow matching
        # p_t(x|x_1) = N(x; mu_t, sigma_t^2I)
        # mu_t = tx_1
        # sigma_t = (1-t) + \sigma_min t
        # v_t(x|x_1) = (\sigma_min-1)/sigm_t (x-mu_t) + x_1
        
        self.velocity_model = velocity_model
        self.sigma_min = sigma_min
        
    def sigma(self, t):
        return (1-t) + self.sigma_min*t
    
    def mu(self, t, x1):
        return t*x1
    
    def sample_from_p_tx_x1(self, t, x1):
        mu = self.mu(t, x1)
        std = self.sigma(t)
        return torch.randn_like(x1) * std + mu
    
    def velocity_conditioned_on_x1(self, t, x, x1):
        sigma_t = self.sigma(t)
        mu_t = self.mu(t, x1)
        return (self.sigma_min-1)/sigma_t*(x-mu_t) + x1
    
    def corruption_process(self, x1, t):
        xt = self.sample_from_p_tx_x1(t, x1)
        return xt
    
    def train_step(self, x1, optimizer, *args, **kwargs):
        optimizer.zero_grad()
        
        t = torch.rand(len(x1)).view(-1, 1).clamp(min=1e-3, max=1e10).to(x1)
        xt = self.sample_from_p_tx_x1(t, x1)
        velocity_label = self.velocity_conditioned_on_x1(t, xt, x1)
        
        loss = ((self.velocity_model(torch.cat([t, xt], dim=-1)) - velocity_label)**2).mean()
        
        loss.backward()
        optimizer.step()
        return {"loss": loss.detach().cpu().item()}
    
    def solve_ode(self, x0, dt=0.01):
        T = int(1/dt)
        xt = copy.deepcopy(x0)
        xtraj = [copy.deepcopy(xt).unsqueeze(1)]
        
        for t in torch.linspace(0, 1, T):
            t = t.clamp(min=1.0e-3, max=1.0e10).to(x0).view(-1, 1).repeat(len(x0), 1)
            velocity_t = self.velocity_model(torch.cat([t, xt], dim=-1))
            xt = xt + velocity_t * dt
            xtraj.append(copy.deepcopy(xt.detach()).unsqueeze(1))
        return xt, torch.cat(xtraj, dim=1)            