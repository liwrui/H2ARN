import torch


def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, 3, N]
        npoint: number of samples
    Return:
        sampled_points: sampled pointcloud, [B, 3, N]
        centroids: sampled pointcloud index, [B, npoint]
    """

    xyz = xyz.transpose(2, 1)  # [B, N, 3]
    device = xyz.device
    B, N, C = xyz.shape
    sampled_points = torch.zeros(B, npoint, 3).to(device)

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    barycenter = torch.sum((xyz), 1)
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]

    for i in range(npoint):
        centroids[:, i] = farthest
        sampled_points[:, i, :] = xyz[batch_indices, farthest, :]
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, -1)[1]

    sampled_points = sampled_points.transpose(2, 1).contiguous()
    return sampled_points, centroids


def fps_1(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        sampled_points: sampled pointcloud, [npoint, 3]
        centroids: sampled pointcloud index, [npoint]
    """
    device = xyz.device
    N, C = xyz.shape
    sampled_points = torch.zeros(npoint, 3).to(device)
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)

    distance = torch.ones(N).to(device) * 1e10

    barycenter = torch.sum(xyz, 0)
    barycenter = barycenter / xyz.shape[0]
    dist = torch.sum((xyz - barycenter) ** 2, dim=1)
    farthest = torch.argmax(dist)

    for i in range(npoint):
        centroids[i] = farthest
        sampled_points[i] = xyz[farthest]
        centroid = xyz[farthest].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)

        distance = torch.min(distance, dist)

        farthest = torch.argmax(distance)

    return sampled_points, centroids
