import torch


def jacobian(y, x, create_graph=False):
    # xx, yy = x.detach().numpy(), y.detach().numpy()
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=True)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def f(xx):
    # y = x * x * torch.arange(4, dtype=torch.float)
    matrix = torch.tensor([[0.2618, 0.2033, 0.7280, 0.8618],
        [0.1299, 0.6498, 0.6675, 0.0527],
        [0.3006, 0.9691, 0.0824, 0.8513],
        [0.7914, 0.2796, 0.3717, 0.9483]], requires_grad=True)
    y = torch.einsum('ji, i -> j', (matrix, xx))
    return y


if __name__ == "__main__":
    # matrix = torch.rand(4, 4, requires_grad=True)
    # print(matrix)
    x = torch.arange(4,  dtype=torch.float, requires_grad=True)
    print(jacobian(f(x), x))
    grad = torch.autograd.functional.jacobian(f, x).numpy()
    # grad = grad.flatten()
    print(grad)
    # print(hessian(f(x, matrix), x))
