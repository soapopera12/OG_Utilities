import torch
import torch.nn as nn
import torch.autograd as autograd

class GradNorm:
    def __init__(self, num_tasks: int, alpha: float, learning_rate_weights: float = 0.025):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks, dtype=torch.float32))
        self.optimizer = torch.optim.Adam([self.weights], lr=learning_rate_weights)
        self.initial_losses = None
        
    def to(self, device):
        self.weights = nn.Parameter(self.weights.to(device))
        self.optimizer = torch.optim.Adam([self.weights], lr=self.optimizer.defaults['lr'])
        return self

    def _get_last_shared_layer(self, model):
        if not hasattr(model, 'shared_base'):
            raise ValueError("Model must have a 'shared_base' attribute to use GradNorm automatically.")
        for layer in reversed(list(model.shared_base.children())):
            if len(list(layer.parameters())) > 0:
                return layer
        raise ValueError("GradNorm could not find a layer with parameters in the model's 'shared_base'.")

    def calculate_losses(self, model, task_losses: list):
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([loss.item() for loss in task_losses], device=self.weights.device)

        last_shared_layer = self._get_last_shared_layer(model)
        
        weighted_grad_norms = []
        for i in range(self.num_tasks):
            grad = autograd.grad(
                self.weights[i] * task_losses[i],
                last_shared_layer.parameters(),
                retain_graph=True,
                create_graph=True
            )[0]
            weighted_grad_norms.append(torch.norm(grad))
        weighted_grad_norms = torch.stack(weighted_grad_norms)

        avg_grad_norm = weighted_grad_norms.mean().detach()

        current_losses = torch.tensor([loss.item() for loss in task_losses], device=self.weights.device)
        loss_ratios = current_losses / self.initial_losses
        inverse_train_rates = loss_ratios / loss_ratios.mean()

        target_grad_norms = avg_grad_norm * (inverse_train_rates ** self.alpha)
        
        grad_norm_loss = torch.abs(weighted_grad_norms - target_grad_norms.detach()).sum()

        total_loss_weighted = (self.weights * torch.stack(task_losses)).sum()

        return total_loss_weighted, grad_norm_loss
        
    def update_weights(self, grad_norm_loss):
        self.optimizer.zero_grad()
        grad_norm_loss.backward()
        self.optimizer.step()
        
    def renormalize_weights(self):
        with torch.no_grad():
            self.weights.data = torch.max(self.weights.data, torch.zeros_like(self.weights.data))
            renorm_coeff = self.num_tasks / self.weights.sum()
            self.weights.data = self.weights.data * renorm_coeff
