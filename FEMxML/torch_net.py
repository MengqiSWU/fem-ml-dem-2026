import torch
from utilSelf.general import echo


class Net_Simple(torch.nn.Module):
    def __init__(self, inputNum, outputNum, device=torch.device('cpu'), layerList='dd', node=20,
                 fourier_features=True, m_freqs=None, activation=torch.nn.Sigmoid()):
        torch.nn.Module.__init__(self)
        self.m_freqs = m_freqs if m_freqs is not None else int(node)
        self.layerList = layerList
        self.device = device
        self.fourier_features = fourier_features
        self.inputNum, self.outputNum = inputNum, outputNum
        self.activation = activation
        if fourier_features:
            self.first = torch.nn.Linear(2 * self.m_freqs + inputNum, node).to(self.device)
            self.B = torch.nn.Linear(inputNum, self.m_freqs).to(self.device)
        else:
            self.first = torch.nn.Linear(inputNum, node, bias=True).to(self.device)
        # NOTE: it is crucial to use a torch.nn.ModuleList to set the parameters trainable
        self.layers = torch.nn.ModuleList(self.getInitLayers(layerList=layerList, node=node, outputNum=outputNum))
        self.model_capacity()

    def getInitLayers(self, layerList, node, outputNum):
        layers = []
        num_layers = len(layerList)
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(torch.nn.Linear(node, outputNum, bias=True).to(self.device))
            else:
                layers.append(torch.nn.Linear(node, node, bias=True).to(self.device))
        return layers

    def model_capacity(self):
        """
        Prints the number of parameters and the number of layers in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_layers = len(list(self.parameters()))
        echo()
        echo("\t\t\tThe number of layers in the model: %d" % num_layers,
             "\n\t\t\tThe number of learnable parameters in the model: %d" % number_of_learnable_params)

    def forward(self, x):
        # x = torch.stack([torch.log(x[:, 0]), x[:, 1]/x[:, 0], x[:, 2]], dim=1)

        # pre-process
        if self.fourier_features:
            x = torch.cat(
                (torch.cos(torch.matmul(x, self.B)), torch.sin(torch.matmul(x, self.B)), x), dim=1)
            x = self.activation(self.first(x))
        else:
            x = self.activation(self.first(x))

        num_layers = len(self.layers)
        i = 0
        while i < (num_layers - 1):
            key = self.layerList[i]
            if key == 'd':
                x = self.activation(self.layers[i](x))
                i += 1
            elif key == 'm':
                x = self.activation(self.layers[i](x * x)) + x
                i += 1
            else:
                raise ValueError('Key cannot be %s' % key)
        x = self.layers[num_layers - 1](x)
        return x

    def get_dy(self, x: torch.Tensor):
        # x = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32, requires_grad=True)
        y = self.forward(x)
        dy = torch.autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones(y.size()).to(self.device),
            create_graph=True)[0]
        return dy


class Net_origin(Net_Simple):
    def __init__(
            self, xmean, xstd, ymean, ystd,
            inputNum, outputNum, device, layerList,
                 node=10,
                 fourier_features=True, m_freqs=None, activation=torch.nn.Sigmoid()
                 ):
        Net_Simple.__init__(
            self, inputNum=inputNum, outputNum=outputNum, device=device,
            layerList=layerList ,node=node, fourier_features=fourier_features,
            m_freqs=m_freqs, activation=activation)
        self.xmean, self.xstd, self.ymean, self.ystd = \
            torch.tensor(xmean, dtype=torch.float, device=device), \
            torch.tensor(xstd, dtype=torch.float, device=device), \
            torch.tensor(ymean, dtype=torch.float, device=device), \
            torch.tensor(ystd, dtype=torch.float, device=device)
        # self.model_capacity()

    def normalization(self, x):
        normed = (x - self.xmean) / self.xstd
        return normed

    def normalization_y(self, y):
        normed = (y - self.ymean) / self.ystd
        return normed

    def re_normalization(self, y):
        re_normed = y * self.ystd + self.ymean
        return re_normed


class Net(Net_origin):
    def __init__(self, xmean, xstd, ymean, ystd,
                 fourier_features, device=torch.device('cpu'),
                 activation=torch.nn.Sigmoid(), inputNum=4, outputNum=1,
                 layerList='dmddmd', node=30):
        Net_origin.__init__(self, xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd,
                            device=device, inputNum=inputNum, outputNum=outputNum, layerList=layerList,
                            activation=activation, node=node, fourier_features=fourier_features)

    def forward(self, x):
        # pre-process
        if self.fourier_features:
            x = torch.cat(
                (torch.cos(self.B(x)), torch.sin(self.B(x)), x), dim=1)
            x = self.activation(self.first(x))
        else:
            x = self.activation(self.first(x))

        num_layers = len(self.layers)
        i = 0
        while i < (num_layers - 1):
            key = self.layerList[i]
            if key == 'd':
                x = torch.relu(self.layers[i](x))
                i += 1
            elif key == 'm':
                x = torch.relu(self.layers[i](x * x)) + x
                i += 1
            else:
                raise ValueError('Key cannot be %s' % key)
        x = self.layers[num_layers - 1](x)
        # reverse process
        return x


class Net_y_dy(Net):
    def __init__(self, xmean, xstd, ymean, ystd,
                 fourier_features, device=torch.device('cpu'),
                 activation=torch.nn.Sigmoid(), inputNum=4, outputNum=1,
                 layerList='dmddmd', node=30):
        Net_origin.__init__(self, xmean=xmean, xstd=xstd, ymean=ymean, ystd=ystd,
                            device=device, inputNum=inputNum, outputNum=outputNum, layerList=layerList,
                            activation=activation, node=node, fourier_features=fourier_features)
        self.layer1 = torch.nn.Linear(self.combination_num, self.outputNum, bias=False)
        # self.a =  torch.autograd.Variable(
        #     torch.randn(self.combination_num, self.outputNum).type(torch.FloatTensor),
        #     requires_grad=True)
        self.model_capacity()

    def forward(self, x):
        x_normed = self.normalization(x=x, xmean=self.xmean, xstd=self.xstd, reverse=False)
        x_combined = self.conbine_2nd_order(x=x_normed)
        x = self.layer1(x_combined)
        q = torch.pow(input=x ** 2, exponent=0.25)
        q = self.normalization(q, xmean=self.ymean, xstd=self.ystd, reverse=True)
        return q

    def conbine_2nd_order(self, x):
        if len(x[0]) == 3:
            x_combined = torch.cat(
                (x[:, 0:1] ** 2, x[:, 2:3] ** 2, x[:, 1:2] ** 2, x[:, 0:1] * x[:, 2:3]), dim=1)
        elif len(x[0]) == 6:
            x_combined = torch.cat(
                (x[:, 0:1] ** 2, x[:, 3:4] ** 2, x[:, 5:6] ** 2,
                 x[:, 0:1] * x[:, 3:4], x[:, 3:4] * x[:, 5:6], x[:, 0:1] * x[:, 5:6],
                 x[:, 1:2] ** 2, x[:, 2:3] ** 2, x[:, 4:5] ** 2), dim=1)
        else:
            echo('Input features of %d has not been added yet, please checked!' % self.inputNum)
            raise
        return x_combined
