import numpy as np

def generate_2d_fluid_flow(
    n_samples=2000,
    grid_size=16,
    time_steps=20,
    viscosity=0.01,
    seed=42
):
    """
    Simple 2D incompressible flow using stream function ψ
    u =  dψ/dy
    v = -dψ/dx
    """
    np.random.seed(seed)

    xs = np.linspace(0, 2*np.pi, grid_size)
    ys = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(xs, ys)

    data = []

    for t in range(time_steps):
        phase = 0.3 * t
        psi = (
            np.sin(X + phase) * np.cos(Y - phase)
            + 0.5 * np.sin(2*X - phase)
        )

        # velocity field
        u = np.gradient(psi, axis=0)
        v = -np.gradient(psi, axis=1)

        flow = np.stack([u, v], axis=-1)  # (H,W,2)
        flow = flow.reshape(-1)           # flatten

        data.append(flow)

    data = np.array(data)

    # add small noise + subsample
    idx = np.random.choice(len(data), n_samples, replace=True)
    data = data[idx]
    data += 0.01 * np.random.randn(*data.shape)

    return data


if __name__ == "__main__":
    flow_data = generate_2d_fluid_flow()
    np.save("flow_like_data.npy", flow_data)
    print("✓ flow_like_data.npy generated:", flow_data.shape)
