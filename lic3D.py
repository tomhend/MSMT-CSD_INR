import numpy as np
import tensorflow as tf

INF = 1e10


# def get_t1(u, f):
#     if u == 0.0:
#         return INF
#     if u > 0:
#         return (1 - f) / u
#     else:
#         return - f / u


# def advance1(ux, uy, fx, fy):
#     '''
#     >>> advance1(1.0, 0.0, 0.6, 0.3)
#     (0.4, 1, 0, 0.0, 0.3)
#     >>> advance1(0.0, 1.0, 0.6, 0.3)
#     (0.7, 0, 1, 0.6, 0.0)
#     '''
#     tx = get_t1(ux, fx)
#     ty = get_t1(uy, fy)

#     if tx < ty:
#         t = tx
#     else:
#         t = ty

#     dx, dy = 0, 0
#     if tx < ty:
#         if ux > 0:
#             dx = +1
#             fx = 0.0
#         else:
#             dx = -1
#             fx = 1.0
#         fy += t * uy
#     else:
#         if uy > 0:
#             dy = +1
#             fy = 0.0
#         else:
#             dy = -1
#             fy = 1.0
#         fx += t * ux

#     return t, dx, dy, fx, fy


def get_t(u, f):
    return tf.where(
        tf.equal(u, 0.0),
        INF * tf.ones_like(u),
        tf.where(
            tf.greater(u, 0),
            (1 - f) / u,
            -f / u
        ))


def advance_3d(ux, uy, uz, fx, fy, fz):
    """
    Advances the streamline one step in 3D.
    
    Returns:
      t      : step length along the streamline.
      dx,dy,dz : integer displacements (±1 or 0) for the grid cell.
      new_fx, new_fy, new_fz : updated fractional positions.
    """

    # Compute time step for each axis.
    tx = get_t(ux, fx)
    ty = get_t(uy, fy)
    tz = get_t(uz, fz)

    # Choose the minimum time step among x, y, and z.
    t = tf.minimum(tf.minimum(tx, ty), tz)

    # Advance fractional positions in all directions.
    new_fx = fx + t * ux
    new_fy = fy + t * uy
    new_fz = fz + t * uz

    one = tf.ones_like(fx)
    zero = tf.zeros_like(fx)

    # Initialize displacements to zero.
    dx = tf.zeros_like(new_fx, dtype=tf.int32)
    dy = tf.zeros_like(new_fy, dtype=tf.int32)
    dz = tf.zeros_like(new_fz, dtype=tf.int32)

    # Determine which axis reached its cell boundary.
    cond_x = tf.logical_and(tf.less_equal(tx, ty), tf.less_equal(tx, tz))
    cond_y = tf.logical_and(tf.less_equal(ty, tx), tf.less_equal(ty, tz))
    cond_z = tf.logical_and(tf.less_equal(tz, tx), tf.less_equal(tz, ty))

    # For the limiting axis, reset its fractional value and assign grid step.
    new_fx = tf.where(
        cond_x,
        tf.where(tf.greater(ux, 0), zero, one),
        new_fx
    )
    dx = tf.where(
        cond_x,
        tf.where(tf.greater(ux, 0), tf.cast(one, tf.int32), -tf.cast(one, tf.int32)),
        dx
    )

    new_fy = tf.where(
        cond_y,
        tf.where(tf.greater(uy, 0), zero, one),
        new_fy
    )
    dy = tf.where(
        cond_y,
        tf.where(tf.greater(uy, 0), tf.cast(one, tf.int32), -tf.cast(one, tf.int32)),
        dy
    )

    new_fz = tf.where(
        cond_z,
        tf.where(tf.greater(uz, 0), zero, one),
        new_fz
    )
    dz = tf.where(
        cond_z,
        tf.where(tf.greater(uz, 0), tf.cast(one, tf.int32), -tf.cast(one, tf.int32)),
        dz
    )

    return t, dx, dy, dz, new_fx, new_fy, new_fz


# def advance_tmax(ux, uy, fx, fy, tmax):
#     t_, dx_, dy_, fx_, fy_ = advance(ux, uy, fx, fy)

#     cond = (t_ < tmax)

#     zero = tf.zeros_like(fy)

#     t = tf.where(cond, t_, zero + tmax)
#     fx = tf.where(cond, fx_, fx + t * ux)
#     fy = tf.where(cond, fy_, fy + t * uy)

#     zero = tf.cast(zero, dtype=tf.int32)

#     dx = tf.where(cond, dx_, zero)
#     dy = tf.where(cond, dy_, zero)

#     return t, dx, dy, fx, fy

def advance_smax_3d(ux, uy, uz, fx, fy, fz, smax):
    """
    Advances the streamline with a maximum path length (smax) in 3D.
    
    Returns:
      t, dx,dy,dz, new_fx,new_fy,new_fz 
    """
    # Compute the magnitude of the vector field.
    u = tf.sqrt(ux**2 + uy**2 + uz**2)
    tmax = smax / u  # maximum allowed time step given smax.

    # Advance one step.
    t_, dx_, dy_, dz_, fx_, fy_, fz_ = advance_3d(ux, uy, uz, fx, fy, fz)
    
    # Check whether the proposed step stays within smax.
    cond = (t_ * u < smax)

    zero = tf.zeros_like(fx)
    t = tf.where(cond, t_, zero + tmax)
    fx = tf.where(cond, fx_, fx + t * ux)
    fy = tf.where(cond, fy_, fy + t * uy)
    fz = tf.where(cond, fz_, fz + t * uz)

    zero_int = tf.cast(zero, dtype=tf.int32)
    dx = tf.where(cond, dx_, zero_int)
    dy = tf.where(cond, dy_, zero_int)
    dz = tf.where(cond, dz_, zero_int)

    return t, dx, dy, dz, fx, fy, fz

def bc_3d(x, y, z, fx, fy, fz, N, M, O):
    """
    Enforces boundary conditions in a 3D volume.
    
    When a coordinate goes out-of-bounds, it is clamped and the fractional
    offset is reset.
    """
    zero = tf.zeros_like(x)
    f_zero = tf.zeros_like(fx)

    # X boundaries.
    cond = (x < 0)
    x = tf.where(cond, zero, x)
    fx = tf.where(cond, f_zero, fx)
    cond = (x >= N)
    x = tf.where(cond, zero + N - 1, x)
    fx = tf.where(cond, f_zero + 1, fx)
    
    # Y boundaries.
    cond = (y < 0)
    y = tf.where(cond, zero, y)
    fy = tf.where(cond, f_zero, fy)
    cond = (y >= M)
    y = tf.where(cond, zero + M - 1, y)
    fy = tf.where(cond, f_zero + 1, fy)
    
    # Z boundaries.
    cond = (z < 0)
    z = tf.where(cond, zero, z)
    fz = tf.where(cond, f_zero, fz)
    cond = (z >= O)
    z = tf.where(cond, zero + O - 1, z)
    fz = tf.where(cond, f_zero + 1, fz)
    
    return x, y, z, fx, fy, fz


# def f(x):
#     return tf.cast(x, tf.float32)


def loop_3d(vx, vy, vz, h, s, t, x, y, z, fx, fy, fz, L, N, M, O, tex, tmax=None, smax=None):
    """
    3D integration loop analogous to the 2D version.
    
    It marches the streamline for L steps (or until stopping criteria)
    and gathers contributions from the texture.
    """
    def cond(i, *args):
        return i < L

    def step(i, h, s, t, x, y, z, fx, fy, fz, pix):
        # Gather vector field and texture values at current voxel.
        xyz = tf.stack([x, y, z], axis=-1)
        ux = tf.gather_nd(vx, xyz)
        uy = tf.gather_nd(vy, xyz)
        uz = tf.gather_nd(vz, xyz)
        p = tf.gather_nd(tex, xyz)
        
        # Use smax branch.
        if smax is not None:
            dt, dx, dy, dz, fx, fy, fz = advance_smax_3d(ux, uy, uz, fx, fy, fz, smax - s)
        else:
            dt, dx, dy, dz, fx, fy, fz = advance_3d(ux, uy, uz, fx, fy, fz)
        
        v = tf.sqrt(ux**2 + uy**2 + uz**2)
        ds = dt * v
        dh = tf.ones_like(ds)  # (Here you could include a weighting function if desired.)
        # dh = ds
        pix = pix + p * dh
        h = h + dh
        s = s + ds
        t = t + dt
        x = x + dx
        y = y + dy
        z = z + dz
        
        # Apply 3D boundary conditions.
        x, y, z, fx, fy, fz = bc_3d(x, y, z, fx, fy, fz, N, M, O)
        i = i + 1
        return i, h, s, t, x, y, z, fx, fy, fz, pix
    
    pix = tf.zeros_like(tex)
    return tf.while_loop(
        cond,
        step,
        [0, h, s, t, x, y, z, fx, fy, fz, pix]
    )

@tf.function(jit_compile=True)
def line_integral_convolution_3d(tex, vx, vy, vz, L, N, M, O, tmax=None, smax=None):
    """
    Performs the LIC on a 3D volume.
    
    Sets up a voxel grid and fractional offsets, then integrates along
    streamlines in both forward and reverse directions.
    """
    shape = tf.shape(tex)
    x = tf.range(shape[0], dtype=tf.int32)
    y = tf.range(shape[1], dtype=tf.int32)
    z = tf.range(shape[2], dtype=tf.int32)
    x, y, z = tf.meshgrid(x, y, z, indexing='ij')
    
    # Initialize fractional positions at the center of each voxel.
    fx = tf.zeros_like(vx, dtype=tf.float32) + 0.5
    fy = tf.zeros_like(vy, dtype=tf.float32) + 0.5
    fz = tf.zeros_like(vz, dtype=tf.float32) + 0.5
    
    h = tf.zeros_like(tex, dtype=tf.float32)
    s = tf.zeros_like(tex, dtype=tf.float32)
    t = tf.zeros_like(tex, dtype=tf.float32)
    
    # Forward integration.
    _, h1, _, _, _, _, _, _, _, _, pix1 = loop_3d(
        vx, vy, vz, h, s, t, x, y, z, fx, fy, fz, L, N, M, O, tex, tmax, smax
    )
    # Reverse integration (integrate along the opposite direction).
    _, h2, _, _, _, _, _, _, _, _, pix2 = loop_3d(
        -vx, -vy, -vz, h, s, t, x, y, z, fx, fy, fz, L, N, M, O, tex, tmax, smax
    )
    
    return (pix1 + pix2) / (h1 + h2)

def runlic_3d(vx, vy, vz, L, magnitude=True, texture=None):
    """
    Generates a random 3D texture and performs LIC on the 3D vector field.
    
    vx, vy, vz : 3D vector field components (numpy arrays with shape (N,M,O)).
    L          : Number of integration steps.
    magnitude  : If True, modulate the output by the vector magnitude.
    """
    assert vx.shape == vy.shape == vz.shape
    N, M, O = vx.shape
    np.random.seed(13)

    if texture is None:
        # seed for reproducibility
        np.random.seed(13)
        tex = np.random.rand(N, M, O)
        # tex = 0.75 * np.random.rand(N, M, O) + 0.25
    else:
        # ensure it has the right shape
        assert texture.shape == (N, M, O)
        tex = texture.astype(np.float32)

    tex_ = tf.constant(tex, dtype=tf.float32)
    vx_ = tf.constant(vx, dtype=tf.float32)
    vy_ = tf.constant(vy, dtype=tf.float32)
    vz_ = tf.constant(vz, dtype=tf.float32)

    mag = tf.sqrt(vx_**2 + vy_**2 + vz_**2) + 1e-12

    ux_ = vx_ / mag
    uy_ = vy_ / mag
    uz_ = vz_ / mag

    tex_out_ = line_integral_convolution_3d(tex_, ux_, uy_, uz_, L, N, M, O)
    if magnitude:
        tex_out_ *= tf.math.erf(tf.sqrt(vx_**2 + vy_**2 + vz_**2))
    
    return tex_out_.numpy()
