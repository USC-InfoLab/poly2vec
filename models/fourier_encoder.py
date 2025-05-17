import torch
import triangle
from shapely import Polygon
import torch.nn as nn


class GeometryFourierEncoder():
    def __init__(self, args, device='cpu'):
        self.device = torch.device(device)
        self.U, self.V = self.create_gfm_meshgrid(args.gfm_params)
        # reshape for batch operations
        self.U = self.U[None, :, :].to(device)
        self.V = self.V[None, :, :].to(device)

    def create_gfm_meshgrid(self, params: dict):
        U = params['n_freqs']  # U is the number of points in the geometric series
        w_min = params['w_min']
        w_max = params['w_max']
        g = (w_max / w_min) ** (1 / (U - 1))  # common ratio

        # Generate the geometric sequence for positive frequencies
        positive_wu = torch.tensor([w_min * g ** u for u in range(U)], dtype=torch.float32)

        # Generate the sets for Wx and Wy according to the definition
        if (2 * U + 1) % 2 == 1:  # If Nux is 2U + 1
            Wx = torch.cat((-torch.flip(positive_wu, dims=[0]), torch.tensor([0]), positive_wu))
        else:  # If Nux is 2U
            Wx = torch.cat((-torch.flip(positive_wu[:-1], dims=[0]), torch.tensor([0]), positive_wu))

        if U % 2 == 1:
            Wy = torch.cat((torch.tensor([0]), positive_wu))
        else:
            Wy = positive_wu

        # Generate the meshgrid for U and V using the geometric frequencies
        U, V = torch.meshgrid(Wx, Wy, indexing='ij')
        return U, V

    def encode(self, geom_dataset: torch.Tensor, lengths: torch.Tensor, dataset_type: str):
        geom_dataset = geom_dataset.to(self.device)

        if dataset_type == "points":
            return self.point_encoder(geom_dataset)

        elif dataset_type == "lines":  # there are no lengths
            return self.line_encoder(geom_dataset)

        elif dataset_type == "polylines":
            return self.polyline_encoder(geom_dataset, lengths)

        elif dataset_type == "polygons":
            return self.polygon_encoder(geom_dataset, lengths)

        else:
            raise ValueError("Invalid input dataset")

    def point_encoder(self, points: torch.Tensor):
        x = points[:, 0].reshape(-1, 1, 1)
        y = points[:, 1].reshape(-1, 1, 1)
        return torch.exp(-2j * torch.pi * (self.U * x + self.V * y))

    def line_encoder(self, lines: torch.Tensor):
        """
        Encode a line dataset
        Args:
            lines: torch.Tensor of shape (n_geoms, 2, 2):
                where 2 is the number of points in the line segment
                and 2 is the x and y coordinates
        """
        lines_expanded = lines.unsqueeze(2).unsqueeze(3)  # shape becomes (n_lines, 2, 1, 1, 2)

        Qx, Qy = lines_expanded[:, 0, :, :, 0], lines_expanded[:, 0, :, :, 1]
        Rx, Ry = lines_expanded[:, 1, :, :, 0], lines_expanded[:, 1, :, :, 1]
        # calculate affine transformation parameters
        det_ = 1 / ((Qx - Rx) ** 2 + (Qy - Ry) ** 2)
        xm = (Qx + Rx) / 2
        ym = (Qy + Ry) / 2
        # compute Fourier transform of the line segments
        return (1.0 / det_) * \
               torch.exp(-2j * torch.pi * (xm * self.U + ym * self.V)) * \
               torch.sinc((Rx - Qx) * self.U + (Ry - Qy) * self.V)

    def polyline_encoder(self, polylines: torch.Tensor, lengths: torch.Tensor):
        """
        Encode a polyline dataset
        Args:
            polylines: torch.Tensor of shape (n_geoms, M, 2):
                where M is the number of points in the polyline padded to the maximum length
            lengths: torch.Tensor of shape (n_geoms,):
                the actual length of each polyline
        """
        encoded = []
        for idx, polyline in enumerate(polylines):
            actual_length = lengths[idx].item()
            if actual_length < 2:  # at least 2 points to form a line
                continue

            FT = torch.zeros_like(self.U, dtype=torch.complex64)
            for i in range(actual_length - 1):
                segment = polyline[i:i + 2]
                segment_ft = self.line_encoder(segment.unsqueeze(0))
                FT += segment_ft.squeeze(0)
            encoded.append(FT)

        return torch.stack(encoded).squeeze()

    def polygon_encoder(self, polygons: torch.Tensor, lengths: torch.Tensor, hole=None):
        encoded = []
        for idx, polygon in enumerate(polygons):
            poly_ft = self.polygon_ft(polygon[:lengths[idx]], hole)
            encoded.append(poly_ft)

        return torch.stack(encoded)

    def polygon_ft(self, polygon: torch.Tensor, holes=None):
        # Preprocess the polygon using Shapely to handle self-intersections and collinear points.
        fixed_polygon = self.preprocess_polygon(polygon.cpu())
        exterior_coords = list(fixed_polygon.exterior.coords)

        # Remember to remove the last closed vertex auto-generated when using Shapely
        triangulation_result = self.cdt_triangulate(exterior_coords[:-1], holes)
        triangles = triangulation_result['vertices'][triangulation_result['triangles']]
        ft_list = []
        for tri in triangles:
            a, b, x0, c, d, y0 = self.triangle_affine(torch.tensor(tri, dtype=torch.float32).to(self.device))
            feature = self.fourier_transform_rtriangle(self.U, self.V, x0, y0, a, b, c, d)
            ft_list.append(feature)
        FT = torch.sum(torch.stack(ft_list, dim=0), dim=0)

        return FT

    def preprocess_polygon(self, polygon_coords):
        """
        Preprocess the polygon using Shapely to handle self-intersections and collinear points.
        :param polygon_coords: List of coordinates defining the exterior of the polygon.
        :return: A Shapely polygon object, preprocessed for triangulation.
        """
        # Create a Shapely polygon
        polygon = Polygon(polygon_coords)

        # Fix any self-intersections and simplify the polygon
        fixed_polygon = polygon.buffer(0)  # Buffering with 0 fixes self-intersections

        return fixed_polygon


    def cdt_triangulate(self, polygon: torch.Tensor, holes=None):
        """
        Performs Constrained Delaunay Triangulation on a polygon with optional holes using the Triangle library.
        :param polygon: List of exterior vertices of the polygon.
        :param holes: List of lists, where each sublist contains vertices defining a hole.
        :return: Triangulation result.
        """
        input_data = {'vertices': polygon, 'segments': [(i, (i + 1) % len(polygon)) for i in range(len(polygon))]}
        if holes:
            input_data['holes'] = holes

        # Perform Constrained Delaunay Triangulation
        triangulation = triangle.triangulate(input_data, 'pYq')

        return triangulation

    def triangle_affine(self, tri, tri_ori=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])):
        """
        Computing the affine transformation matrix T from tri to tri_ori
        """
        array_ori = torch.vstack([tri_ori.T.to(self.device), torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device)])
        array = torch.vstack([tri.T.to(self.device), torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(self.device)])
        T = torch.mm(array_ori, torch.linalg.inv(array))
        a = T[0][0]
        b = T[0][1]
        x0 = T[0][2]
        c = T[1][0]
        d = T[1][1]
        y0 = T[1][2]

        return a, b, x0, c, d, y0

    def fourier_transform_rtriangle(self, U, V, x0=0.0, y0=0.0, a=1.0, b=0.0, c=0.0, d=1.0):
        pi = torch.pi

        det = a * d - b * c

        area = 1.0 / (2.0 * abs(det))

        x = (1.0 / det) * (b * y0 - d * x0)
        y = (1.0 / det) * (c * x0 - a * y0)

        phase_shift = torch.exp(-2j * pi * (U * x + V * y))

        U_ = (U * d - V * c) / det
        V_ = (V * a - U * b) / det

        mask = (U_ + V_ == 0)
        zero_mask = (U_ == 0) & (V_ == 0)
        u_mask = (U_ == 0)
        v_mask = (V_ == 0)

        base_u = torch.exp(-2j * pi * U_)
        base_v = torch.exp(-2j * pi * V_)
        base_uv = torch.exp(-2j * pi * (U_ + V_))

        # general case
        part1_all = 1 / (4 * pi ** 2 * U_ * V_ * (U_ + V_))
        part2_all = U_ * (-base_uv) + (U_ + V_) * base_u - V_

        # special case: u + v == 0
        part1_sp = -1 / (4 * pi ** 2 * U_ ** 2)
        part2_sp = base_u + 2j * pi * U_ - 1
        # Combine using the mask
        part1 = torch.where(mask, part1_sp, part1_all)
        part2 = torch.where(mask, part2_sp, part2_all)

        # special case: v == 0
        part1_sp = 1 / (4 * pi ** 2 * U_ ** 2)
        part2_sp = (2j * pi * U_ + 1) * base_u - 1
        # Combine using the mask
        part1 = torch.where(v_mask, part1_sp, part1)
        part2 = torch.where(v_mask, part2_sp, part2)

        # special case: u == 0
        part1_sp = -1 / (4 * pi ** 2 * V_ ** 2)
        part2_sp = base_v + 2j * pi * V_ - 1
        # Combine using the mask
        part1 = torch.where(u_mask, part1_sp, part1)
        part2 = torch.where(u_mask, part2_sp, part2)

        FT = torch.where(zero_mask, area, (1.0 / abs(det)) * part1 * part2 * phase_shift)

        return FT
