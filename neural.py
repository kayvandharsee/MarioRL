from torch import nn

class MarioNet(nn.Module):
    '''
    Double Deep Q-Learning Network
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        channels, height, width = input_dim
        # Ensure dimensions are correct
        if height != 84:
            raise ValueError(f"Expecting input height: 84, got: {height}")
        if width != 84:
            raise ValueError(f"Expecting input width: 84, got {width}")
        # Build Q_online and Q_target
        self.online = self.__build_network(channels=channels, output_dim=output_dim)
        self.target = self.__build_network(channels=channels, output_dim=output_dim)
        # Copy parameters from Q_online to Q_target
        self.target.load_state_dict(self.online.state_dict())
        # Freeze parameters on Q_target
        for param in self.target.parameters():
            param.requires_grad = False
        
    def conv_dimensions(self, input_dim, kernel_size, stride):
        '''
        Returns the value of height and width (one value to represent both, as it is square)
        after passing through a convolutional layer. We assume zero padding and dilation
        '''
        return (input_dim - kernel_size) // stride + 1
        
    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
    def __build_network(self, channels, output_dim):
        # Get dimensions after all conv layers
        dim = self.conv_dimensions(input_dim=84, kernel_size=8, stride=4)
        dim = self.conv_dimensions(input_dim=dim, kernel_size=4, stride=2)
        dim = self.conv_dimensions(input_dim=dim, kernel_size=3, stride=1)

        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * dim * dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )