class Block(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type = None,
                 k_group = None,
                 dim=None,
                 dt_rank = None,
                 d_inner = None,
                 d_state = None,
                 bimamba=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type

        # in proj ============================
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.conv3d = nn.Conv3d(
            in_channels=d_inner, out_channels=d_inner, groups=d_inner,
            bias = True, kernel_size=(1, 1, 1), ** kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def flatten_spectral_spatial(self, x):
        x = rearrange(x, 'b c t h w -> b c (h w) t')  # [10, 192, 64, 28]
        x = rearrange(x, 'b c n m -> b c (n m)')  # [10, 192, 1792]
        return x
    
    def flatten_spatial_spectral(self, x):
        x = rearrange(x, 'b c t h w -> b c t (h w)')  # [10, 192, 28, 64]
        x = rearrange(x, 'b c n m -> b c (n m)')  # [10, 192, 1792]
        return x
    
    def reshape_spectral_spatial(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.view(B, H * W, T, -1)
        y = rearrange(y, 'b o t c -> b t o c')
        y = y.view(B, T, H, W, -1)
        return y
    
    def reshape_spatial_spectral(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.view(B, T, H, W, -1)
        return y

    def scan(self, x, scan_type=None, group_type=None):
        """Fixed scan method with proper error handling and default case"""
        if scan_type == 'Spectral-priority':
            x = self.flatten_spectral_spatial(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == 'Spatial-priority':
            x = self.flatten_spatial_spectral(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == 'Cross spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spa, dims=[-1])], dim=1)
        elif scan_type == 'Cross spatial-spectral':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spa, torch.flip(x_spe, dims=[-1])], dim=1)
        elif scan_type == 'Parallel spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spe, dims=[-1]), x_spa, torch.flip(x_spa, dims=[-1])], dim=1)
        else:
            # Default case: if scan_type is None or not recognized, use spectral-priority as default
            print(f"Warning: Unknown scan_type '{scan_type}', using 'Spectral-priority' as default")
            x = self.flatten_spectral_spatial(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        
        return xs

    def forward(self, x: Tensor, SelectiveScan = SelectiveScanMamba):
        x = self.in_proj(x)  # d_inner=192  [10, 64, 96] -> [10, 64, 384]    [10, 8, 8, 96]->[10, 8, 8, 384]
        x, z = x.chunk(2, dim=-1)  # [10, 64, 192]  [10, 8, 8, 192]
        z = self.act(z)  # [10, 64, 192]   [10, 8, 8, 192]

        # forward con1d
        x = x.permute(0, 4, 1, 2, 3).contiguous() #[64, 192, 28, 8, 8]
        x = self.conv3d(x) #[64, 192, 28, 8, 8]
        x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, T, H, W = x.shape
        L = T * H * W
        D, N = self.A_logs.shape  # D 768   N 16
        K, D, R = self.dt_projs_weight.shape  # 4   192    6

        # scan - now properly handles all cases
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight) #[10, 2, 64, 64] einsum指定输入张量和输出张量之间的维度关系，你可以定义所需的运算操作
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)  #[10, 2, 32, 64]  [10, 2, 16, 64]  [10, 2, 16, 64]
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)  #[10, 2, 192, 64]

        xs = xs.view(B, -1, L) # [10, 384, 64]  [10, 768, 64]
        dts = dts.contiguous().view(B, -1, L) # [10, 768, 64] .contiguous()是一个用于确保张量存储连续性
        Bs = Bs.contiguous()  #[10, 2, 16, 64]   [10, 4, 16, 64]
        Cs = Cs.contiguous()  #[10, 2, 16, 64]   [10, 4, 16, 64]

        As = -torch.exp(self.A_logs.float())   # [384, 16]  [768, 16]
        Ds = self.Ds.float() # (k * d)  [384]  [768]
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # [384]  [768]

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)  #[10, 384, 64]->[10, 2, 192, 64]    [10, 4, 192, 64]
        assert out_y.dtype == torch.float

        if self.group_type == 'Cube':
            if self.scan_type == 'Spectral-priority' or self.scan_type is None:
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [10, 192, 64]
                y = self.reshape_spectral_spatial(y, B, H, W, T)
                y = self.out_norm(y)  # [10, 64, 192]
            elif self.scan_type == 'Spatial-priority':
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [10, 192, 64]
                y = self.reshape_spatial_spectral(y, B, H, W, T)
                y = self.out_norm(y)  # [10, 64, 192]
            elif self.scan_type == 'Cross spectral-spatial':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spectral_spatial(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spatial_spectral(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Cross spatial-spectral':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spatial_spectral(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spectral_spatial(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Parallel spectral-spatial':
                ye = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [10, 192, 64]
                ye = self.reshape_spectral_spatial(ye, B, H, W, T)
                ya = out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1])  # [10, 192, 64]
                ya = self.reshape_spatial_spectral(ya, B, H, W, T)
                y = ye + ya
                y = self.out_norm(y)  # [10, 64, 192]
        else:
            # Default handling if group_type is not 'Cube'
            print(f"Warning: Unknown group_type '{self.group_type}', using default processing")
            y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])  # [10, 192, 64]
            y = self.reshape_spectral_spatial(y, B, H, W, T)
            y = self.out_norm(y)  # [10, 64, 192]

        y = y * z   #[10, 64, 192]   [10, 8, 8, 192]
        out = self.dropout(self.out_proj(y))  #[10, 64, 96]   [10, 8, 8, 96]

        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)