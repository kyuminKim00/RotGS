import torch
import torch.nn as nn
from utils.general_utils import get_expon_lr_func
import matplotlib.pyplot as plt
from utils.system_utils import searchForMaxIteration
import os
import numpy as np

class ResidualPredictor(nn.Module):
    def __init__(self, number_of_cameras=1, num_ctrl_points=1, device="cuda"):
        super().__init__()
        self.num_ctrl_points = num_ctrl_points
        self.residuals = nn.Parameter(
            torch.zeros(number_of_cameras, num_ctrl_points + 1, device=device)
        )
        ctrl_positions = torch.linspace(0, 1, num_ctrl_points + 1, device=device)
        self.register_buffer("ctrl_positions", ctrl_positions)
        self.optimizer = None

    def train_setting(self, training_args):
        l = [
            {'params': self.parameters(),
             'lr': training_args.residual_lr_init,
             "name": "residual"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.rotpredictor_scheduler_args = get_expon_lr_func(lr_init=training_args.residual_lr_init,
                                                       lr_final=training_args.residual_lr_final,
                                                       max_steps=training_args.residual_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "residual":
                lr = self.rotpredictor_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def plot_residual(self, output_folder='.'):
        residuals = np.degrees(self.residuals[0].detach().cpu().numpy())
        ctrl_positions = self.ctrl_positions.detach().cpu().numpy()

        plt.figure(figsize=(10, 5))
        plt.plot(ctrl_positions, -residuals, marker='o', linestyle='-', color='r')
        plt.xlabel('Control Point (Normalized Time)')
        plt.ylabel('Residual Value (degree)')
        plt.title('Residuals over Control Points')
        plt.savefig(f"{output_folder}/graph/residual.png")
        plt.close()

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "residual_predictor/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'residual_predictor.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "residual_predictor"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "residual_predictor/iteration_{}/residual_predictor.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))

    def forward(self, time: torch.Tensor, cam_idx:int):
        segment_idx = torch.bucketize(time, self.ctrl_positions[1:], right=False)

        t0 = self.ctrl_positions[segment_idx]
        t1 = self.ctrl_positions[segment_idx + 1]

        r0 = self.residuals[cam_idx, segment_idx]
        r1 = self.residuals[cam_idx, segment_idx + 1]

        alpha = (time - t0) / (t1 - t0 + 1e-8)
        residual_t = (1 - alpha) * r0 + alpha * r1

        return residual_t

    
