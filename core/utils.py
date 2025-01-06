import torch

class EMA:
    def __init__(self, beta: float):
        """
        Initialize the EMA class with a beta value.

        Args:
            beta (float): The smoothing factor for the exponential moving average.
        """
        super().__init__()
        self.beta: float = beta
        self.step: int = 0

    def update_model_average(self, ma_model, curr_model):
        """
        Update the parameters of the moving average model (ma_model)
        using the parameters of the current model (curr_model).

        Args:
            ma_model: The moving average model.
            curr_model: The current model providing the new parameters.
        """
        for curr_params, ma_params in zip(curr_model.parameters(), ma_model.parameters()):
            ma_params.data = self.update_avg(ma_params.data, curr_params.data)

    def update_avg(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """
        Compute the updated moving average.

        Args:
            old (torch.Tensor): The current moving average value.
            new (torch.Tensor): The new value to incorporate.

        Returns:
            torch.Tensor: The updated moving average value.
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema: int = 2000):
        """
        Perform one step of EMA updates.

        Args:
            ema_model: The model to update with EMA.
            model: The source model providing the new parameter values.
            step_start_ema (int, optional): The step after which EMA updates begin.
                Defaults to 2000.
        """
        self.step += 1  # Increment step count at the start
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)

    def reset_parameters(self, ema_model, model):
        """
        Reset the parameters of the EMA model to match the source model.

        Args:
            ema_model: The model whose parameters will be reset.
            model: The source model providing the parameter values.
        """
        ema_model.load_state_dict(model.state_dict())
