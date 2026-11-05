#!/usr/bin/env python
# coding: utf-8

# # ISSM helper
# 
# Functions to run ISSM step by step from Python, with the model open to changes between steps. Each
# solve runs ISSM's own code, and without changes the results are identical to ISSM's native transient
# (`solve(md, 'Transient')` with a moving front). The calving laws use ISSM's own level-set calving,
# and the solves go to a persistent ISSM process that is built automatically the first time.
# 
# The top half holds everything you would call; the bottom half is bookkeeping.
# 
# | function | what it does |
# |---|---|
# | `initialize_model(md, ...)` | sets up everything that does not change during a run |
# | `dirichlet(md, (mask, value), ...)` | builds Dirichlet (spc) fields |
# | `rate_factor(T)` | the Arrhenius rate factor A(T) behind `temperature` |
# | `terminus_levelset(md, terminus, surface)` | the ice level set from a terminus line (e.g. digitized from imagery) |
# | `order_boundary(md)`, `plot_boundary(md, i = ..., j = ...)` | ordered boundary nodes, for choosing boundary segments |
# | `interpolate_raster(md, raster)` | a raster (e.g. a GeoTIFF) on the mesh |
# | `diagnostic_solve(md)` | stress balance: returns the velocity |
# | `prognostic_solve(md, Δt)` | one step of front and thickness with the current velocity |
# | `coupled_solve(md, Δt)` | both in one call (`num_steps = n` for several steps) |
# | `set_thickness(md, H)` | a new thickness, with the geometry in hydrostatic equilibrium |
# | `rate_law(md, rate = ...)` | calving at a rate (m/a) |
# | `position_law(md, forcing = ..., threshold = ...)` | calving where a forcing exceeds a threshold |
# | `plot_field(md, field)` | a field on the ice, with the terminus |
# | `ice_volume(md)` | the total ice volume |
# | `effective_strain_rate(md)` | from the last stress balance |
# | `flowline_mesh(md, x)` | a flowline (1D) model: see *Flowline models* |
# | `valley_mesh(md, x, y)` | a structured plan-view mesh with vertices at given positions along and across a valley |
# | `front_diagnostics(md)` | ice area, and the length of the front and the flux and mean speed through it |
# | `profile(md, field)`, `plot_profile(md, field)`, `plot_geometry(md)` | fields and the geometry along a flowline |
# | `terminus_position(md)`, `grounding_line_position(md)` | where the ice ends, and where it starts to float, along a flowline |
# 
# A typical loop:
# 
# ```python
# md = initialize_model(md, thickness = H_0, ice_levelset = x - x_t, spcvx = ..., spcvy = ..., spcthickness = ...)
# 
# for step in tqdm.trange(num_steps):
#     u = diagnostic_solve(md)
#     rate = 0.5*np.hypot(u[0], u[1]) #m/a
#     rate_law(md, rate = rate, extrapolate = True)
#     H, s = prognostic_solve(md, Δt)
# ```

# In[ ]:


######################
### ISSM functions ###
######################

from calving import calving
from frictionweertman import frictionweertman
from ismodelselfconsistent import ismodelselfconsistent
from loadresultsfromdisk import loadresultsfromdisk
from marshall import marshall
from project3d import project3d
from setflowequation import setflowequation
from squaremesh import squaremesh
from iluasmoptions import iluasmoptions
from solve import solve

#######################
### python packages ###
#######################

import atexit, copy, inspect, io, re, subprocess, tempfile
import os, sys, time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from types import SimpleNamespace


# # Setting up a model
# 
# `initialize_model` does most of the work; the other functions here help build its inputs.

# ## `initialize_model`
# 
# Set up everything that is unlikely to change during a simulation. Only `thickness` is required;
# all fields can be scalars or nodal arrays on the 2D mesh.
# 
# | keyword | meaning | default |
# |---|---|---|
# | `thickness` | ice thickness (m) | required |
# | `approximation` | `'SSA'` or `'HO'` (HO extrudes the mesh) | `'SSA'` |
# | `temperature` | ice temperature (K), sets the rheology via `rate_factor` | 273.15 |
# | `ice_levelset` | <0 ice, >0 no ice, 0 on the terminus (signed distance recommended) | all ice |
# | `bed` | bed elevation (m) | none: deep water, fully floating |
# | `waterline` | water level (sea or lake) (m) | 0 |
# | `rho_water` | water density (kg m^-3), e.g. 1000 for a lake | ISSM default (1023) |
# | `friction`, `m` | Weertman C and m, in ISSM's convention τ_b = C² \|u\|^(1/m-1) u (C ~ 10³ is realistic for m = 3) | 0, 3 |
# | `spcvx`, `spcvy` | velocity Dirichlet values (NaN = free), see `dirichlet` | none |
# | `spcthickness` | thickness Dirichlet values (NaN = free) | none |
# | `smb`, `basal_melt` | surface mass balance, floating-ice basal melt (m/a) | 0, 0 |
# | `velocity` | initial velocity guess `(vx, vy)` | 0 |
# | `moving_front` | evolve `ice_levelset` with the ice (ISSM's level-set method); see `rate_law` for calving | True |
# | `grounding_line` | grounding-line migration | True if `bed` is given |
# | `kill_icebergs` | remove ice not connected to grounded ice | True only if some ice is grounded |
# | `reinit_frequency` | steps between level-set reinitializations | ISSM default (10) |
# | `requested_outputs` | extra stress-balance outputs | strain rates |
# | `vertical_layers`, `extrusion_exponent` | HO only | 5, 1 |
# 
# On a mesh from `flowline_mesh`, it also sets up the flowline (see *Flowline models*).
# 
# Without MUMPS, ISSM solves its linear systems with GMRES but stops after 100 iterations, too few for
# HO models (the solve fails with 'Recovery solver failed'). If the model still has that default,
# `initialize_model` allows up to 1000 iterations, stopping once the relative residual is below 1e-10.
# On a single process, the stress balance instead uses PETSc's own direct solver, since GMRES can fail
# on it altogether (e.g. with long, thin elements).
# 
# Returns the model (extrusion for HO creates new mesh fields, so use `md = initialize_model(md, ...)`).
# 
# `kill_icebergs` must stay off for an entirely floating shelf: ISSM only keeps ice connected to fully
# grounded elements, so it would delete the whole shelf on the first step.

# In[ ]:


def initialize_model(md, **kwargs):

    if 'thickness' not in kwargs:
        raise RuntimeError('initialize_model: thickness is required.')
    if hasattr(md.mesh, 'x2d'):
        raise RuntimeError('initialize_model expects a 2D mesh (it extrudes the mesh itself for HO).')

    approx = kwargs.get('approximation', 'SSA').upper()
    if approx not in ['SSA', 'HO']:
        raise RuntimeError('approximation must be "SSA" or "HO".')

    nv, ne = md.mesh.numberofvertices, md.mesh.numberofelements
    vfield = lambda v: np.asarray(v, dtype = float)*np.ones(nv)

    md.miscellaneous.name = kwargs.get('name', md.miscellaneous.name or 'model')
    if 'rho_water' in kwargs:
        md.materials.rho_water = kwargs['rho_water']
    if 'min_thickness' in kwargs:
        md.masstransport.min_thickness = kwargs['min_thickness']
    ϱ = md.materials.rho_ice/md.materials.rho_water

    #################
    ### levelsets ###
    #################

    ice_levelset = vfield(kwargs.get('ice_levelset', -1.))
    md.mask.ice_levelset = ice_levelset

    ################
    ### geometry ###
    ################

    sealevel = vfield(kwargs.get('waterline', 0.))
    has_bed = kwargs.get('bed', None) is not None
    bed = vfield(kwargs['bed']) if has_bed else sealevel - 1e5 #no bed: deep water everywhere

    H = vfield(kwargs['thickness'])
    H[ice_levelset > 0] = 0 #no ice outside the ice domain...
    H = np.maximum(H, md.masstransport.min_thickness) #...which ISSM represents with the minimum thickness

    if 'ocean_levelset' in kwargs:
        md.mask.ocean_levelset = vfield(kwargs['ocean_levelset'])
    else:
        md.mask.ocean_levelset = H + (bed - sealevel)/ϱ #ISSM's floatation criterion
    grounded = md.mask.ocean_levelset > 0

    base = np.where(grounded, bed, sealevel - ϱ*H)
    md.geometry.bed = bed
    md.geometry.thickness = H
    md.geometry.base = base
    md.geometry.surface = base + H

    ################
    ### rheology ###
    ################

    T = kwargs.get('temperature', 273.15)
    md.materials.rheology_n = 3*np.ones(ne)
    md.materials.rheology_B = vfield(rate_factor(T)**(-1/3))

    ################
    ### friction ###
    ################

    md.friction = frictionweertman()
    md.friction.C = vfield(kwargs.get('friction', 0.))
    md.friction.m = kwargs.get('m', 3)*np.ones(ne)

    ###########################
    ### boundary conditions ###
    ###########################

    md.stressbalance.spcvx = vfield(kwargs.get('spcvx', np.nan))
    md.stressbalance.spcvy = vfield(kwargs.get('spcvy', np.nan))
    if getattr(md.miscellaneous, 'flowline', False):
        md.stressbalance.spcvy = vfield(0.) #flowline: no lateral flow anywhere
    md.stressbalance.spcvz = vfield(np.nan)
    md.stressbalance.referential = np.full((nv, 6), np.nan)
    md.stressbalance.loadingforce = np.zeros((nv, 3))
    md.masstransport.spcthickness = vfield(kwargs.get('spcthickness', np.nan))

    ######################################
    ### forcings and initial velocity ###
    ######################################

    md.smb.mass_balance = vfield(kwargs.get('smb', 0.))
    md.basalforcings.groundedice_melting_rate = vfield(0.)
    md.basalforcings.floatingice_melting_rate = vfield(kwargs.get('basal_melt', 0.))

    u0 = kwargs.get('velocity', (0., 0.))
    md.initialization.vx = vfield(u0[0])
    md.initialization.vy = vfield(u0[1])
    md.initialization.vz = vfield(0.)
    md.initialization.pressure = vfield(0.)

    ####################
    ### moving front ###
    ####################

    md.levelset.spclevelset = vfield(np.nan)
    md.levelset.kill_icebergs = int(kwargs.get('kill_icebergs', np.any(grounded)))
    if 'reinit_frequency' in kwargs:
        md.levelset.reinit_frequency = kwargs['reinit_frequency']
    md.calving.calvingrate = vfield(0.) #required by ISSM when the front moves; zero = no calving
    md.frontalforcings.meltingrate = vfield(0.) #likewise; zero = no frontal melt

    ###############
    ### outputs ###
    ###############

    outputs = ['StrainRatexx', 'StrainRateyy', 'StrainRatexy']
    if approx == 'HO':
        outputs += ['StrainRatexz', 'StrainRateyz', 'StrainRatezz']
    outputs = kwargs.get('requested_outputs', outputs)
    md.stressbalance.requested_outputs = ['default'] + [o for o in outputs if o != 'default']

    ################################################
    ### extrude if necessary, then flow equation ###
    ################################################

    mesh2d = copy.deepcopy(md.mesh)
    sigma = None
    if approx == 'HO':
        with quiet_output(): #ISSM's frictionweertman prints a leftover debug line when extruded
            md = md.extrude(kwargs.get('vertical_layers', 5), kwargs.get('extrusion_exponent', 1))
        md.initialization.sealevel = _to3d(md, sealevel) #every layer (the default extrusion only fills the base layer)
        sigma = (md.mesh.z - md.geometry.base)/md.geometry.thickness #relative height; constant, as in ISSM
    else:
        md.initialization.sealevel = sealevel
    md = setflowequation(md, approx, 'all')
    if getattr(md.miscellaneous, 'flowline', False): #pair the two edges of the strip, so every field is the same across it
        md.stressbalance.vertex_pairing = _flowline_pairs(md)
        md.masstransport.vertex_pairing = md.stressbalance.vertex_pairing
        md.settings.solver_residue_threshold = 1e-3 #ISSM pairs with penalties, which make the linear systems less well conditioned

    md.timestepping.start_time = kwargs.get('start_time', 0.)
    md.miscellaneous.helper = SimpleNamespace(
        is3d = (approx == 'HO'),
        moving_front = bool(kwargs.get('moving_front', True)),
        grounding_line = bool(kwargs.get('grounding_line', has_bed)),
        step = 0, #steps completed (ISSM restarts its own counter at every solve)
        sigma = sigma,
        mesh2d = mesh2d,
        laplacian = None,
        last_velocity = None,
        adjacency = None, #mesh connectivity (position_law)
        pending_calving = None, #nodes position_law will calve in the next step
        front = None, #the two stress-balance solutions of an interpolated front (diagnostic_solve)
        checked = set(), #solution types already checked for consistency
    )
    if dict(md.toolkits.DefaultAnalysis) == dict(iluasmoptions()): #ISSM's default without MUMPS (see above)
        md.toolkits.DefaultAnalysis = iluasmoptions('ksp_max_it', 1000, 'ksp_rtol', 1e-10)
        md.toolkits.RecoveryAnalysis = md.toolkits.DefaultAnalysis
        if md.cluster.np == 1: #PETSc's direct solver is serial only
            md.toolkits.addoptions('StressbalanceAnalysis', {'toolkit': 'petsc', 'mat_type': 'aij', 'ksp_type': 'preonly', 'pc_type': 'lu'})

    md.miscellaneous.outputs = SimpleNamespace()
    md.miscellaneous.calving_flux = [] #(time, total calving flux in Gt/a) per step while calving is on
    md.miscellaneous.calving_events = [] #(time, area in m², volume in m³) per position_law event

    _get_worker(md) #build the ISSM worker if needed (first time only) and start it

    return md


# ## `dirichlet`
# 
# Build a Dirichlet (spc) field: NaN where the value is free, the prescribed value elsewhere.
# Pairs `(mask, value)` are applied in order, so later pairs win where masks overlap (e.g. at
# corners). `value` can be a scalar or a nodal array.
# 
# Example: `spcvx = dirichlet(md, (Γ_inflow, 500), (Γ_sidewalls, 0))`

# In[ ]:


def dirichlet(md, *pairs):

    field = np.full(md.mesh.numberofvertices, np.nan)
    for mask, value in pairs:
        mask = np.asarray(mask, dtype = bool)
        field[mask] = np.asarray(value, dtype = float)[mask] if np.ndim(value) else value
    return field


# ## `rate_factor`
# 
# Arrhenius rate factor A(T) (s^-1 Pa^-3) in the Cuffey & Paterson (2010) form, used by
# `initialize_model` to set the rheology from temperature.

# In[ ]:


def rate_factor(T):

    T = np.asarray(T, dtype = float)
    T_t = 263.15 #transition temperature (K)
    Q = np.where(T < T_t, 6e4, 115e3) #activation energy (J mol^-1)
    A_0 = 3.5e-25 #rate prefactor (s^-1 Pa^-3)
    R = 8.314 #gas constant (J mol^-1 K^-1)
    return A_0*np.exp(-Q/R*(1/T - 1/T_t))


# ## `terminus_levelset`
# 
# The ice level set (signed distance to the terminus, negative on the ice) from a terminus line in
# mesh coordinates, such as one digitized from satellite imagery. The line does not need to reach the
# domain boundary: both ends are extended straight out until they leave the domain, and the extended
# line splits the domain in two. `surface` (e.g. a DEM on the mesh) tells the sides apart: the ice is
# on the side with the higher mean surface, the water on the other. Call it on the 2D mesh, before
# `initialize_model`.

# In[ ]:


def terminus_levelset(md, terminus, surface):

    from shapely import LineString, Polygon, contains_xy, distance, points
    from shapely.ops import split

    x, y = md.mesh.x, md.mesh.y
    bnd = order_boundary(md)
    domain = Polygon(np.c_[x[bnd], y[bnd]]).buffer(1) #1 m buffer, so boundary nodes count as inside

    #extend each end straight out to where it first leaves the domain
    c = np.asarray(terminus.coords)
    reach = np.hypot(np.ptp(x), np.ptp(y))
    ends = []
    for end, previous in [(c[0], c[1]), (c[-1], c[-2])]:
        direction = (end - previous)/np.linalg.norm(end - previous)
        exits = LineString([end, end + reach*direction]).intersection(domain.exterior)
        exit_points = np.array([g.coords[0] for g in getattr(exits, 'geoms', [exits]) if not g.is_empty])
        if contains_xy(domain, *end) and exit_points.size:
            first = exit_points[np.argmin(np.hypot(*(exit_points - end).T))]
            ends.append(first + direction) #1 m beyond the boundary
        else:
            ends.append(end)
    line = LineString(np.vstack([ends[0], c, ends[1]]))

    #the side with the lowest mean surface is water, the rest ice
    parts = sorted(split(domain, line).geoms, key = lambda part: np.mean(surface[contains_xy(part, x, y)]))
    water = contains_xy(parts[0], x, y)
    return np.where(water, 1, -1)*distance(points(x, y), line)


# ## `order_boundary`
# 
# Sometimes it is necessary to order boundary nodes by looping from one end of a terminus to the
# other along the glacier boundary, e.g. to pick a contiguous section as a named boundary for
# imposing Dirichlet BCs.

# In[ ]:


def order_boundary(md, **kwargs):

    mesh2d = md.miscellaneous.helper.mesh2d if hasattr(md.miscellaneous, 'helper') else md.mesh
    x, y = mesh2d.x, mesh2d.y
    bnd = np.asarray(mesh2d.segments[:, :2], dtype = int) - 1

    adj = {}
    for i, j in bnd:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    nodes = np.fromiter(adj.keys(), int)

    # pick a deterministic start: leftmost, then lowest
    start = nodes[np.lexsort((y[nodes], x[nodes]))][0]

    path = [start]
    prev, cur = -1, start
    while True:
        nxts = adj[cur]
        nxt = nxts[0] if nxts[0] != prev else nxts[1]
        if nxt == start:
            break
        path.append(nxt)
        prev, cur = cur, nxt

    return np.asarray(path)


# ## `plot_boundary`
# 
# Plot a subset of the (ordered) boundary from node `i` to node `j`, for choosing the segments for a
# given boundary condition.

# In[ ]:


def plot_boundary(md, **kwargs):

    index_1 = kwargs.get('i', 0)
    index_2 = kwargs.get('j', None)

    figsize = kwargs.get('figsize', (8, 8))
    linewidth = kwargs.get('linewidth', 0.2)
    color = kwargs.get('color', 'k')
    lw = kwargs.get('lw', 2)
    num_ticks = kwargs.get('num_ticks', 50)
    fontsize = kwargs.get('fontsize', 12)

    mesh2d = md.miscellaneous.helper.mesh2d if hasattr(md.miscellaneous, 'helper') else md.mesh
    x, y, tri = mesh2d.x, mesh2d.y, np.asarray(mesh2d.elements, dtype = int) - 1

    bnd_nodes = order_boundary(md)
    subset = bnd_nodes[index_1:index_2]

    fig, ax = plt.subplots(figsize = figsize)
    ax.triplot(x, y, tri, linewidth = linewidth)
    ax.plot(x[subset], y[subset], lw = lw, color = color)

    for k in range(0, len(bnd_nodes), num_ticks):
        ax.text(x[bnd_nodes[k]], y[bnd_nodes[k]], str(k), fontsize = fontsize)

    ax.set_aspect('equal')
    plt.show()


# ## `interpolate_raster`
# 
# Raster to mesh gridpoints.

# In[ ]:


def interpolate_raster(md, raster, **kwargs):

    import rasterio
    from rasterio.fill import fillnodata
    from rasterio.warp import transform
    from InterpFromGridToMesh import InterpFromGridToMesh

    mesh = md.miscellaneous.helper.mesh2d if hasattr(md.miscellaneous, 'helper') else md.mesh
    x, y = mesh.x, mesh.y

    fill = kwargs.get('fill', True)
    epsg = kwargs.get('epsg', 32645)
    mesh_crs = f'EPSG:{epsg}'

    with rasterio.open(raster) as src:

        z = src.read(1).astype(float)
        t = src.transform
        ny, nx = z.shape
        src_crs = src.crs

        if fill:
            z = fillnodata(z, mask = np.isfinite(z))

        cols = np.arange(nx) + 0.5
        rows = np.arange(ny) + 0.5

        xs = t.c + cols*t.a
        ys = t.f + rows*t.e

        if ys[1] < ys[0]:
            ys = ys[::-1]
            z = z[::-1, :]

        # Transform mesh → raster CRS if needed
        if src_crs is not None and str(src_crs) != mesh_crs:
            x, y = transform(mesh_crs, src_crs, x.tolist(), y.tolist())
            x = np.asarray(x)
            y = np.asarray(y)

    vals = InterpFromGridToMesh(xs, ys, z, x, y, np.nan)
    return vals


# # Running the model
# 
# Each solve runs ISSM's own code for one time step (or, with `coupled_solve(..., num_steps = n)`,
# several) and copies ISSM's end-of-step state back into `md`. Between calls, change anything on `md`
# directly, or pass one of these keywords to any of the solves:
# 
# | keyword | effect |
# |---|---|
# | `velocity` | velocity (initial guess for the stress balance; advecting velocity for `prognostic_solve`) |
# | `thickness` | thickness, via `set_thickness` (hydrostatic geometry) |
# | `thickness_inflow` | values on the nodes constrained by `spcthickness` |
# | `friction` | Weertman coefficient C |
# | `temperature` | rheology via `rate_factor` |
# | `smb` | surface mass balance |
# | `ice_levelset` | ice domain (e.g. a manual calving event) |
# 
# Passing back exactly what a solve returned changes nothing, so the icepack-style
# `u, H, s = coupled_solve(md, Δt, velocity = u, thickness = H)` works too. All solves also accept
# `quiet = False` (show ISSM's output) and `check = True` (run ISSM's consistency check on every call).

# ## `diagnostic_solve`
# 
# Mirrors `icepack`'s function of the same name: solves the stress balance for the current state
# (ISSM's native stress balance) and returns the velocity `u = (vx, vy)`, or `(vx, vy, vz)` for HO.
# Values outside the ice are zero.
# 
# **Fronts inside elements.** With a moving front, the front usually runs through elements. ISSM then
# counts the whole of each such element as ice, with the thickness of its ice-free vertices, but
# applies the water pressure only along the part of the front in elements with a single ice vertex:
# the stress balance behaves as if the ice reached the far side of the elements the front cuts, and
# jumps whenever the front crosses a row of vertices (in a test glacier, the terminus speed jumped by
# 4–8 m/a at every crossing on a 5 m mesh). `diagnostic_solve` therefore solves the stress balance
# twice while the front cuts elements: with the front on the vertices just behind it (those elements
# ice-free) and on the vertices just ahead of it (those elements full of ice), both of which ISSM
# handles exactly, and interpolates the velocity linearly between them by how far the front has
# crossed each element (along its edges). The velocity then varies smoothly as the front moves
# through an element and passes continuously from one pair of rows to the next (residual bumps
# of about 1 m/a on that 5 m mesh). The level set, calving and mass transport are ISSM's own, with the
# true front. `front_interpolation = False` gives ISSM's single native solve; with a fixed front, or
# a front on vertices, there is only one solve anyway. The two solutions are kept for
# `front_diagnostics`. `coupled_solve` always uses ISSM's native stress balance.

# In[ ]:


def diagnostic_solve(md, **kwargs):

    st = _state(md)
    _apply_tweaks(md, kwargs)
    quiet, check = kwargs.get('quiet', True), kwargs.get('check', None)

    configurations = None
    if st.moving_front and kwargs.get('front_interpolation', True):
        configurations = _front_configurations(md)

    st.front = None
    if configurations is None:
        u, sol = _stressbalance(md, quiet, check)
        _store_outputs(md, sol)
    else:
        behind, ahead, θ = configurations
        φ, u0 = np.array(md.mask.ice_levelset, dtype = float), _velocity(md)
        try:
            md.mask.ice_levelset = _to3d(md, behind)
            u_behind, sol_behind = _stressbalance(md, quiet, check)
            _set_velocity(md, u0)
            md.mask.ice_levelset = _to3d(md, ahead)
            u_ahead, sol_ahead = _stressbalance(md, quiet, check)
        finally:
            md.mask.ice_levelset = φ
        u_carried = extrapolate(md, u_behind, ice_levelset = _to3d(md, behind)) #the solution behind, carried past its front
        w = _to3d(md, θ)
        u = tuple((1 - w)*b + w*a for b, a in zip(u_carried, u_ahead))
        _store_outputs(md, _blend_outputs(md, sol_behind, sol_ahead, w))
        st.front = SimpleNamespace(behind = behind, ahead = ahead, weights = θ,
                                   velocity_behind = u_behind, velocity_ahead = u_ahead)

    _set_velocity(md, u)
    st.last_velocity = _velocity(md)
    return st.last_velocity

def _stressbalance(md, quiet, check):

    out = solve_quiet(md, 'Stressbalance', quiet, check)
    sol = out.results.StressbalanceSolution
    md.results = out.results
    u = (sol.Vx, sol.Vy) + ((sol.Vz,) if _state(md).is3d else ())
    return tuple(np.asarray(ui, dtype = float).ravel() for ui in u), sol


# ## `prognostic_solve`
# 
# Mirrors `icepack`'s function of the same name, but returns both thickness and surface elevation
# `(H, s)`. Advances the model by one time step with the current velocity (usually from
# `diagnostic_solve`): the ice front moves (if `moving_front`), then thickness is transported, as in
# ISSM's native transient. `diagnostic_solve` followed by `prognostic_solve` is one native step.

# In[ ]:


def prognostic_solve(md, Δt, **kwargs):

    st = _state(md)
    _apply_tweaks(md, kwargs)

    if st.moving_front:
        _set_velocity(md, extrapolate(md, _velocity(md)))
    _run(md, Δt, 1, False, kwargs.get('quiet', True), kwargs.get('check', None))

    return _to2d(md, md.geometry.thickness), _to2d(md, md.geometry.surface)


# ## `coupled_solve`
# 
# Stress balance and prognostic step in one ISSM call: identical to `diagnostic_solve` followed by
# `prognostic_solve`, but cheaper. Returns `(u, H, s)`, where the dimension of `u` depends on the
# approximation; with a moving front, `u` outside the ice holds ISSM's extrapolated velocity.
# 
# `num_steps` runs several native steps in a single call (no tweaks possible in between), which is
# as fast as ISSM's own transient. `md.results` holds the output of the last ISSM call (with
# `num_steps`, those steps; at most one earlier call is used to keep the reinitialization aligned).

# In[ ]:


def coupled_solve(md, Δt, **kwargs):

    st = _state(md)
    _apply_tweaks(md, kwargs)

    _run(md, Δt, int(kwargs.get('num_steps', 1)), True, kwargs.get('quiet', True), kwargs.get('check', None))

    st.last_velocity = _velocity(md)
    return st.last_velocity, _to2d(md, md.geometry.thickness), _to2d(md, md.geometry.surface)


# ## `set_thickness`
# 
# Set the ice thickness and put the geometry in hydrostatic equilibrium, as ISSM does: grounded
# where ISSM's floatation level set `H + (bed - sea level)·ρw/ρi` is positive (only re-evaluated
# when grounding-line migration is on), floating elsewhere. Thickness is floored at
# `md.masstransport.min_thickness`. Accepts 2D or 3D fields.

# In[ ]:


def set_thickness(md, H):

    st = _state(md)
    ϱ = md.materials.rho_ice/md.materials.rho_water
    H = np.maximum(_to3d(md, H), md.masstransport.min_thickness)
    bed = md.geometry.bed
    sealevel = md.initialization.sealevel

    if st.grounding_line:
        md.mask.ocean_levelset = H + (bed - sealevel)/ϱ
    grounded = md.mask.ocean_levelset > 0

    base = np.where(grounded, bed, sealevel - ϱ*H)
    md.geometry.thickness = H
    md.geometry.base = base
    md.geometry.surface = base + H
    if st.is3d:
        md.mesh.z = base + st.sigma*H


# # Calving
# 
# Both laws use ISSM's level-set calving, so they need `moving_front = True` (the default).

# ## `rate_law`
# 
# A rate-based calving law: the ice front retreats, relative to the ice flow, at `rate` metres per
# year perpendicular to itself (a scalar or a nodal field). This is ISSM's own level-set calving:
# the rate becomes `md.calving.calvingrate`, and in every moving-front step ISSM moves the front with
# the ice velocity minus the rate along the front's outward normal, so flow and calving act together
# within each step. The rate persists until it is set again; `rate_law(md, 0)` switches calving off.
# 
# - Calving stops where the bed is at or above the waterline, so the front cannot retreat onto land.
#   The level set resolves this to within about one element: where the rate drops from its full
#   value to zero across a single element, the front comes to rest up to about one element past the
#   shoreline (a rate that falls off towards the shore, such as a water-depth law, avoids the jump).
#   `water_only = False` removes this cap.
# - `extrapolate = True` keeps the rate on the ice (every vertex of an element containing ice) and
#   extends it outward the way ISSM extends the velocity, so the front's motion depends only on the
#   rate inside the ice (the level-set analog of sampling just inside the terminus). Use it for rates
#   computed from fields that are only meaningful in the ice, such as stresses, and set the rate
#   again every step. Leave it off for fields that are valid everywhere, such as water depth.
# - For a rate that depends on the current velocity or stresses, set it between `diagnostic_solve`
#   and `prognostic_solve`: it then uses the same step's velocity, as ISSM's built-in laws do. Set
#   before `coupled_solve`, it lags by one step.
# - While the rate is nonzero, ISSM's total calving flux (`TotalCalvingFluxLevelset`, converted to
#   Gt/a) is recorded after every step in `md.miscellaneous.calving_flux` as `(time, flux)` pairs.
# 
# Returns the calving rate as applied (2D field).

# In[ ]:


def rate_law(md, rate, **kwargs):

    st = _state(md)
    if not st.moving_front:
        raise RuntimeError('rate_law needs a moving front: initialize_model(md, ..., moving_front = True).')

    c = np.maximum(_to3d(md, rate), 0.) #ISSM requires calving rates >= 0

    if kwargs.get('extrapolate', False):
        c = extrapolate(md, c)
    if kwargs.get('water_only', True):
        c[md.geometry.bed >= md.initialization.sealevel] = 0.

    if type(md.calving).__name__ != 'calving': #ISSM's default law: a prescribed rate, perpendicular to the front
        md.calving = calving()
    md.calving.calvingrate = c

    return _to2d(md, c)


# ## `position_law`
# 
# A position-based calving law: ice calves where `forcing` (a nodal field, e.g. a stress or a
# crevasse depth) exceeds `threshold`. It works like ISSM's own discrete calving laws (height above
# buoyancy, crevasse depth): it marks the nodes that must become ice-free, and in the next front
# update ISSM's level-set solve pins them there, after which ISSM removes the ice and carries on. The
# level set is reinitialized right after that step, so it stays a signed distance to the new front.
# 
# - `search_mode = 'local'` (default): calving starts at the ice front and spreads through connected
#   nodes above the threshold, stopping where the forcing drops below it, as ISSM's crevasse-depth law
#   spreads from the front. For mechanisms acting on the terminus itself.
# - `search_mode = 'nonlocal'`: in addition, a region above the threshold further upstream calves,
#   together with all the ice seaward of it, when it cuts that ice off from the grounded glacier (or,
#   for a fully floating shelf, from the inflow, i.e. the nodes with a thickness constraint). A rift
#   has to span the glacier before it releases an iceberg; a partial one does nothing.
# - `max_search_distance` (m, default unlimited): only nodes within this distance behind the front
#   can start calving; in local mode this caps how far the front jumps back, in nonlocal mode it is
#   how far upstream a region may begin. Distances come from the level set itself.
# - Only below the waterline (`water_only = False` removes this), never on nodes with a thickness
#   constraint (the inflow), and any ice cut off by the calving goes with it. The front lands on the calved nodes, so it is resolved to within an element, as with
#   ISSM's own discrete laws. A 3D forcing is read on the base layer.
# - The calving happens in the next `prognostic_solve` or `coupled_solve`: called between
#   `diagnostic_solve` and `prognostic_solve` it has ISSM's own timing; called before `coupled_solve`
#   it applies after that step's stress balance.
# - Each call that calves records `(time, area, volume)` of the ice to be calved (m², m³) in
#   `md.miscellaneous.calving_events`.
# 
# Returns the mask of nodes to be calved (2D).

# In[ ]:


def position_law(md, forcing, threshold, **kwargs):

    st = _state(md)
    if not st.moving_front:
        raise RuntimeError('position_law needs a moving front: initialize_model(md, ..., moving_front = True).')
    mode = kwargs.get('search_mode', 'local')
    if mode not in ['local', 'nonlocal']:
        raise RuntimeError("position_law: search_mode must be 'local' or 'nonlocal'.")
    reach = kwargs.get('max_search_distance', None)
    reach = np.inf if reach is None else reach

    x, y, elts, nv2 = _mesh2d(md)
    A = _adjacency(md)
    levelset = _to2d(md, md.mask.ice_levelset)
    f = np.asarray(forcing, dtype = float).ravel()
    f = f[0]*np.ones(nv2) if f.size == 1 else _to2d(md, f)

    ice = levelset < 0
    distance = -levelset #distance behind the front
    front = np.zeros(nv2, dtype = bool) #vertices of the elements the front runs through
    front[elts[np.any(levelset[elts] < 0, axis = 1) & np.any(levelset[elts] >= 0, axis = 1)].ravel()] = True

    exceed = ice & (f > threshold) & ~np.isfinite(_to2d(md, md.masstransport.spcthickness)) #never calve the inflow
    if kwargs.get('water_only', True):
        exceed &= _to2d(md, md.geometry.bed) < _to2d(md, md.initialization.sealevel)

    #local: spread from the front through connected nodes above the threshold
    region = exceed & (distance <= reach)
    labels = _components(A, region)
    calved = region & np.isin(labels, labels[region & front])

    #nonlocal: whole regions starting within reach calve if they cut ice off
    if mode == 'nonlocal':
        labels = _components(A, exceed)
        candidate = np.isin(labels, labels[exceed & (distance <= reach)]) & ~calved
        cut = _cut_off(md, ice & ~calved & ~candidate)
        cutting = np.isin(labels, labels[candidate & ((A @ cut.astype(float)) > 0)])
        calved |= candidate & cutting

    calved |= _cut_off(md, ice & ~calved) #ice cut off by the calving goes with it

    if calved.any():
        if not (ice & ~calved).any():
            raise RuntimeError('position_law: this would calve all the ice.')
        #ice removed per element: ice fraction before minus after the calved nodes are pinned (+1)
        removed = (_ice_fraction(levelset, elts) - _ice_fraction(np.where(calved, 1., levelset), elts))*_element_areas(x, y, elts)
        H = _to2d(md, md.geometry.thickness)
        md.miscellaneous.calving_events.append((float(md.timestepping.start_time),
                                                float(removed.sum()), float(np.sum(removed*H[elts].mean(axis = 1)))))
        pending = getattr(st, 'pending_calving', None)
        st.pending_calving = calved if pending is None else (pending | calved)

    return calved


# ## `front_diagnostics`
# 
# The state of the ice front after the last `diagnostic_solve`, as a namespace:
# 
# | field | meaning |
# |---|---|
# | `area` | ice area (m²), from the level set (sub-element) |
# | `length` | length of the front (m) |
# | `flux` | ice flux through the front (m³/a): thickness times the depth-averaged velocity normal to the front, integrated along it |
# | `speed` | mean depth-averaged speed normal to the front (m/a), weighted by length |
# 
# The front is the zero contour of `ice_levelset`, wherever it runs (the calving front, but also any
# lateral ice margin). When `diagnostic_solve` interpolated between the two fronts on vertices, the
# flux, speed and length are those of the two solutions at their own fronts, interpolated by the
# fraction of the band between them that is ice, so they too vary smoothly with the front position.
# For a front through the full width of a valley, `area/width` is the mean glacier length and
# `flux/width` the flux per unit width. Requires a 2D (plan-view) or flowline mesh.

# In[ ]:


def front_diagnostics(md):

    st = _state(md)
    x, y, elts, nv2 = _mesh2d(md)
    φ = _to2d(md, md.mask.ice_levelset)
    H = _to2d(md, md.geometry.thickness)
    area = float(np.sum(_ice_fraction(φ, elts)*_element_areas(x, y, elts)))

    def along_front(levelset, u):
        ū = [_depth_average(md, ui) for ui in u[:2]]
        return _contour_integrals(x, y, elts, levelset, H, ū)

    front = getattr(st, 'front', None)
    if front is None:
        u = st.last_velocity if st.last_velocity is not None else _velocity(md)
        length, flux, normal_speed = along_front(φ, u)
    else:
        #ice area of the two fronts on vertices: no element is cut, so the elements containing ice
        a_b, a_a = [float(np.sum((np.min(ls[elts], axis = 1) < 0)*_element_areas(x, y, elts))) for ls in (front.behind, front.ahead)]
        θ = 0. if a_a <= a_b else float(np.clip((area - a_b)/(a_a - a_b), 0, 1))
        b = along_front(front.behind, front.velocity_behind)
        a = along_front(front.ahead, front.velocity_ahead)
        length, flux, normal_speed = [(1 - θ)*vb + θ*va for vb, va in zip(b, a)]
    return SimpleNamespace(area = area, length = length, flux = flux, speed = normal_speed/length if length > 0 else np.nan)


# ## `valley_mesh`
# 
# A structured plan-view mesh with rows of vertices at positions `x` (along the valley) and `y`
# (across it), at any spacing: ISSM's `squaremesh` with its vertices moved onto those positions.
# Useful for idealized valley glaciers, with fine spacing where the front moves.

# In[ ]:


def valley_mesh(md, x, y):

    x, y = np.sort(np.asarray(x, dtype = float)), np.sort(np.asarray(y, dtype = float))
    md = squaremesh(md, x[-1] - x[0], y[-1] - y[0], x.size, y.size)
    i = np.rint((md.mesh.x - md.mesh.x.min())/(md.mesh.x.max() - md.mesh.x.min())*(x.size - 1)).astype(int)
    j = np.rint((md.mesh.y - md.mesh.y.min())/(md.mesh.y.max() - md.mesh.y.min())*(y.size - 1)).astype(int)
    md.mesh.x, md.mesh.y = x[i], y[j]
    return md


# # Looking at results

# ## `plot_field`
# 
# Plot a nodal field on the 2D mesh, hiding triangles with no ice and drawing the terminus (zero
# contour of `ice_levelset`). For extruded meshes, 3D fields are shown at the surface layer.

# In[ ]:


def plot_field(md, field, ax = None, **kwargs):

    label = kwargs.pop('label', None)
    title = kwargs.pop('title', None)
    front = kwargs.pop('front', True)
    levels = kwargs.pop('levels', 20)

    x, y, elts, nv2 = _mesh2d(md)
    field = np.asarray(field, dtype = float).ravel()
    field = field[-nv2:] if field.size > nv2 else field
    levelset = _to2d(md, md.mask.ice_levelset)

    ax = plt.gca() if ax is None else ax
    tri = mtri.Triangulation(x, y, elts)
    tri.set_mask(np.all(levelset[elts] > 0, axis = 1))
    c = ax.tricontourf(tri, field, levels = levels, **kwargs)
    ax.figure.colorbar(c, ax = ax, label = label)
    if front:
        ax.tricontour(x, y, elts, levelset, levels = [0], colors = 'k', linestyles = 'dashed')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)
    ax.grid()
    return c


# ## `ice_volume`
# 
# Total ice volume (m³): for every element, the area covered by ice (from the level set, which is
# linear on each element) times the element's mean thickness.

# In[ ]:


def ice_volume(md):

    x, y, elts, nv2 = _mesh2d(md)
    frac = _ice_fraction(_to2d(md, md.mask.ice_levelset), elts)
    H = _to2d(md, md.geometry.thickness)
    return float(np.sum(frac*_element_areas(x, y, elts)*H[elts].mean(axis = 1)))


# ## `effective_strain_rate`
# 
# Effective strain rate from the strain-rate components of the last stress balance (requested by
# default in `initialize_model`).

# In[ ]:


def effective_strain_rate(md, **kwargs):

    from DepthAverage import DepthAverage

    depth_averaged = kwargs.get('depth_averaged', False) if md.flowequation.isHO else False #SSA is already depth-averaged
    out = md.miscellaneous.outputs
    ε_xx, ε_xy, ε_yy = out.StrainRatexx, out.StrainRatexy, out.StrainRateyy

    inner_ε = ε_xx**2 + 2*ε_xy**2 + ε_yy**2
    trace_ε = ε_xx + ε_yy

    if md.flowequation.isHO:
        ε_xz, ε_yz, ε_zz = out.StrainRatexz, out.StrainRateyz, out.StrainRatezz
        inner_ε += ε_zz**2 + 2*ε_xz**2 + 2*ε_yz**2
        trace_ε += ε_zz

    ε_e = np.sqrt(1/2*(inner_ε + trace_ε**2)) #generally true since the trace term should be zero in 3D
    ε_e = ε_e if not depth_averaged else DepthAverage(md, ε_e)

    return np.asarray(ε_e).flatten()


# # Flowline models
# 
# For effectively 1D problems: a strip two nodes wide along x, with no lateral flow and every field
# the same across it, so ISSM solves a flowline model with all of the machinery above (the solves, the
# moving front, both calving laws), for SSA and HO alike. The side walls exert no drag. Build the mesh
# with `flowline_mesh` and compute fields from `md.mesh.x`; `initialize_model` recognizes the strip and
# sets up the rest: no lateral velocity, and the two edges of the strip paired, so that ISSM keeps every
# field identical across it. The glacier flows toward +x.

# ## `flowline_mesh`
# 
# A strip two nodes wide, with nodes at positions `x` along the flowline (any spacing). `width` defaults
# to the mean node spacing, which keeps the elements well shaped; it has no effect on the results.

# In[ ]:


def flowline_mesh(md, x, **kwargs):

    x = np.sort(np.asarray(x, dtype = float))
    width = kwargs.get('width', np.mean(np.diff(x)))
    md = squaremesh(md, x[-1] - x[0], width, x.size, 2)
    column = np.rint(md.mesh.x/(x[-1] - x[0])*(x.size - 1)).astype(int)
    md.mesh.x = x[column]
    md.miscellaneous.flowline = True
    return md


# ## `profile`
# 
# A field along the flowline, as `(x, values)` sorted by x. 3D fields are taken at the surface layer.

# In[ ]:


def profile(md, field):

    x, y, elts, nv2 = _mesh2d(md)
    row = np.flatnonzero(y == y.min())
    row = row[np.argsort(x[row])]
    field = np.asarray(field, dtype = float).ravel()
    field = field[-nv2:] if field.size > nv2 else field
    return x[row], field[row]


# ## `plot_profile`
# 
# Plot a field along the flowline where there is ice, with the terminus dashed: the flowline
# counterpart of `plot_field`.

# In[ ]:


def plot_profile(md, field, ax = None, **kwargs):

    label = kwargs.pop('label', None)
    title = kwargs.pop('title', None)

    x, f = profile(md, field)
    ice = profile(md, md.mask.ice_levelset)[1] < 0

    ax = plt.gca() if ax is None else ax
    lines = ax.plot(x, np.where(ice, f, np.nan), **kwargs)
    ax.axvline(terminus_position(md), color = 'k', linestyle = 'dashed', linewidth = 1)
    ax.set_xlabel('x (m)')
    ax.set_ylabel(label)
    if title is not None:
        ax.set_title(title)
    ax.grid()
    return lines


# ## `plot_geometry`
# 
# The longitudinal section: bedrock, water up to the waterline, and ice between its base and surface,
# with the terminus dashed and the grounding line marked on the bed.

# In[ ]:


def plot_geometry(md, ax = None, **kwargs):

    title = kwargs.pop('title', None)

    x, bed = profile(md, md.geometry.bed)
    base, surface, sealevel = (profile(md, f)[1] for f in [md.geometry.base, md.geometry.surface, md.initialization.sealevel])
    ice = profile(md, md.mask.ice_levelset)[1] < 0
    floor = bed.min() - 0.05*(surface.max() - bed.min())

    ax = plt.gca() if ax is None else ax
    ax.fill_between(x, floor, bed, color = '0.55', linewidth = 0)
    ax.fill_between(x, bed, sealevel, where = bed < sealevel, color = 'lightsteelblue', linewidth = 0, interpolate = True)
    ax.fill_between(x, base, surface, where = ice, facecolor = 'aliceblue', edgecolor = 'k', linewidth = 0.8)
    ax.axvline(terminus_position(md), color = 'k', linestyle = 'dashed', linewidth = 1)
    x_gl = grounding_line_position(md)
    if np.isfinite(x_gl):
        ax.plot(x_gl, np.interp(x_gl, x, bed), 'k^', markersize = 7)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(floor, None)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('elevation (m)')
    if title is not None:
        ax.set_title(title)


# ## `terminus_position`, `grounding_line_position`
# 
# Where the ice ends, and where it starts to float, along the flowline (m), from the level sets.

# In[ ]:


def terminus_position(md):

    return _crossing(md, md.mask.ice_levelset)

def grounding_line_position(md):

    return _crossing(md, -md.mask.ocean_levelset)


# # Bookkeeping
# 
# Nothing below needs to be called directly: it is how the functions above work.

# ## Mesh and field utilities
# 
# Shared by the functions above. Extruded (3D) meshes store vertices layer by layer, base layer first,
# so 2D fields are the first `numberofvertices2d` entries.

# In[ ]:


def _state(md):

    st = getattr(md.miscellaneous, 'helper', None)
    if st is None:
        raise RuntimeError('Call initialize_model(md, ...) first.')
    return st

def _mesh2d(md):

    m = md.mesh
    if hasattr(m, 'x2d'):
        return m.x2d, m.y2d, np.asarray(m.elements2d, dtype = int) - 1, int(m.numberofvertices2d)
    return m.x, m.y, np.asarray(m.elements, dtype = int) - 1, int(m.numberofvertices)

def _to2d(md, field):

    field = np.asarray(field, dtype = float).ravel()
    nv2 = _mesh2d(md)[3]
    return field[:nv2] if field.size > nv2 else field

def _to3d(md, field):

    nv = md.mesh.numberofvertices
    field = np.asarray(field, dtype = float).ravel()
    if field.size == 1:
        return field[0]*np.ones(nv)
    if field.size == nv:
        return field.copy()
    return project3d(md, 'vector', field, 'type', 'node').ravel()

def _velocity(md):

    u = (md.initialization.vx, md.initialization.vy)
    if _state(md).is3d:
        u = u + (md.initialization.vz,)
    return tuple(np.asarray(ui, dtype = float).ravel().copy() for ui in u)

def _set_velocity(md, u):

    md.initialization.vx = _to3d(md, u[0])
    md.initialization.vy = _to3d(md, u[1])
    if _state(md).is3d:
        md.initialization.vz = _to3d(md, u[2]) if len(u) > 2 else np.zeros(md.mesh.numberofvertices)


# ## Stepping
# 
# `_run` performs `num_steps` native transient steps and copies ISSM's end-of-step state back into
# `md` (geometry, level sets, velocity, vertex positions, clock). ISSM restarts its step counter at
# every solve, and reinitializes the level set when `step % reinit_frequency == 0`; `_schedule`
# splits the steps into at most two ISSM calls whose reinitialization frequency makes this happen
# at the same global steps as in a single native run. A calving event from `position_law` is
# applied by a first ISSM call of one step, with the calved nodes pinned ice-free in the level-set
# solve and the level set reinitialized right after.
# 
# `_apply_tweaks` applies the solve keywords listed under *Running the model*.

# In[ ]:


def _schedule(done, n, N):

    if N <= 0:
        yield n, 0
        return
    m = (-done) % N #steps until the next multiple of N
    if m:
        if n < m:
            yield n, 0
            return
        yield m, m
        n -= m
    if n:
        yield n, N

def _update_state(md, sol, stressbalance):

    st = _state(md)
    get = lambda name: np.asarray(getattr(sol, name), dtype = float).ravel()

    md.geometry.thickness = get('Thickness')
    md.geometry.base = get('Base')
    md.geometry.surface = get('Surface')
    if st.moving_front:
        md.mask.ice_levelset = get('MaskIceLevelset')
    if st.grounding_line:
        md.mask.ocean_levelset = get('MaskOceanLevelset')
    if st.is3d:
        md.mesh.z = md.geometry.base + st.sigma*md.geometry.thickness

    if stressbalance:
        #with a moving front these are already extrapolated outside the ice, exactly the field ISSM carries
        md.initialization.vx = get('Vx')
        md.initialization.vy = get('Vy')
        if st.is3d:
            md.initialization.vz = get('Vz')
        _store_outputs(md, sol)

def _store_outputs(md, sol):

    requested = [o for o in md.stressbalance.requested_outputs if o != 'default']
    md.miscellaneous.outputs = SimpleNamespace(**{name: np.asarray(getattr(sol, name)).ravel()
                                                  for name in requested if hasattr(sol, name)})

def _run(md, Δt, num_steps, stressbalance, quiet, check):

    st = _state(md)
    N = int(md.levelset.reinit_frequency)
    spc = md.levelset.spclevelset

    md.transient.deactivateall()
    md.transient.isstressbalance = int(stressbalance)
    md.transient.ismasstransport = 1
    md.transient.issmb = 1
    md.transient.ismovingfront = int(st.moving_front)
    md.transient.isgroundingline = int(st.grounding_line)
    record_calving = st.moving_front and np.any(np.asarray(md.calving.calvingrate) > 0)
    md.transient.requested_outputs = ['default'] + (['TotalCalvingFluxLevelset'] if record_calving else [])
    md.settings.output_frequency = 1
    md.timestepping.time_step = Δt

    calls, done = [], st.step
    calved = getattr(st, 'pending_calving', None) if st.moving_front else None
    if calved is not None: #one step with the calved nodes pinned ice-free, then reinitialize
        calls.append((1, 1, calved))
        done, num_steps = done + 1, num_steps - 1
    if num_steps > 0:
        calls += [(n, reinit, None) for n, reinit in _schedule(done, num_steps, N if st.moving_front else 0)]

    try:
        for n, reinit, pinned in calls:
            md.levelset.reinit_frequency = reinit
            md.levelset.spclevelset = spc if pinned is None else np.where(_to3d(md, pinned) > 0, 1., spc)
            md.timestepping.final_time = md.timestepping.start_time + n*Δt
            out = solve_quiet(md, 'Transient', quiet, check)
            steps = out.results.TransientSolution
            _update_state(md, steps[-1], stressbalance)
            if record_calving:
                for k in range(len(steps)):
                    flux = float(np.squeeze(steps[k].TotalCalvingFluxLevelset))*md.constants.yts/1e12 #kg/s -> Gt/a
                    md.miscellaneous.calving_flux.append((float(np.squeeze(steps[k].time)), flux))
            if pinned is not None:
                st.pending_calving = None
            md.results = out.results
            st.step += n
            md.timestepping.start_time = md.timestepping.final_time
    finally:
        md.levelset.reinit_frequency = N
        md.levelset.spclevelset = spc

def _apply_tweaks(md, kwargs):

    st = _state(md)

    u = kwargs.get('velocity', None)
    returned = st.last_velocity #passing back exactly what a solve returned is not a tweak
    if u is not None and not (returned is not None and len(u) == len(returned)
                               and all(np.array_equal(np.ravel(a), np.ravel(b)) for a, b in zip(u, returned))):
        _set_velocity(md, u)

    H = kwargs.get('thickness', None)
    if H is not None and not np.array_equal(_to2d(md, H), _to2d(md, md.geometry.thickness)):
        set_thickness(md, H)

    if kwargs.get('thickness_inflow', None) is not None:
        Γ = np.isfinite(md.masstransport.spcthickness)
        md.masstransport.spcthickness[Γ] = _to3d(md, kwargs['thickness_inflow'])[Γ]

    if kwargs.get('friction', None) is not None:
        md.friction.C = _to3d(md, kwargs['friction'])
    if kwargs.get('temperature', None) is not None:
        md.materials.rheology_B = _to3d(md, rate_factor(kwargs['temperature'])**(-1/3))
    if kwargs.get('smb', None) is not None:
        md.smb.mass_balance = _to3d(md, kwargs['smb'])
    if kwargs.get('ice_levelset', None) is not None:
        md.mask.ice_levelset = _to3d(md, kwargs['ice_levelset'])


# ## `extrapolate`
# 
# Extend nodal fields from the ice into the ice-free area exactly as ISSM's `ExtrapolationAnalysis`
# does inside the moving-front step: values are kept on every vertex of an element that contains
# ice, and the P1 Laplace equation is solved on all other vertices. On extruded meshes this is done
# layer by layer, which is what ISSM's 3D formulation reduces to.
# 
# ISSM performs this step itself for thickness, and for velocity whenever the stress balance runs
# in the same step. `prognostic_solve` runs without the stress balance, so it calls this for the
# velocity. Accepts one array or a tuple of arrays.

# In[ ]:


def _laplacian(md):

    st = _state(md)
    if st.laplacian is None:
        import scipy.sparse as sp
        x, y, elts, nv2 = _mesh2d(md)
        xe, ye = x[elts], y[elts]
        b = np.roll(ye, -1, axis = 1) - np.roll(ye, -2, axis = 1)
        c = np.roll(xe, -2, axis = 1) - np.roll(xe, -1, axis = 1)
        twice_area = np.abs(np.sum(xe*b, axis = 1))
        Ke = (b[:, :, None]*b[:, None, :] + c[:, :, None]*c[:, None, :])/(2*twice_area)[:, None, None]
        rows = np.repeat(elts, 3, axis = 1).ravel()
        cols = np.tile(elts, (1, 3)).ravel()
        st.laplacian = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape = (nv2, nv2)).tocsr()
    return st.laplacian

def extrapolate(md, fields, **kwargs):

    from scipy.sparse.linalg import splu

    single = isinstance(fields, np.ndarray)
    fields = [fields] if single else list(fields)

    x, y, elts, nv2 = _mesh2d(md)
    levelset = _to2d(md, kwargs.get('ice_levelset', md.mask.ice_levelset))
    active = np.zeros(nv2, dtype = bool)
    active[elts[np.any(levelset[elts] < 0, axis = 1)].ravel()] = True
    free = ~active
    if not free.any() or not active.any():
        return fields[0] if single else tuple(fields)

    K = _laplacian(md)
    K_ff = K[free][:, free].tocsc()
    K_fa = K[free][:, active]
    try:
        lu = splu(K_ff)
    except RuntimeError: #ice-free patch with no ice around it: regularize so it stays solvable
        warnings.warn('extrapolate: ice-free area disconnected from the ice; regularizing.')
        import scipy.sparse as sp
        lu = splu((K_ff + 1e-12*abs(K_ff).max()*sp.identity(K_ff.shape[0])).tocsc())

    out = []
    for f in fields:
        F = np.array(f, dtype = float).reshape(-1, nv2) #(layers, vertices)
        F[:, free] = lu.solve(-(K_fa @ F[:, active].T)).T
        out.append(F.ravel())
    return out[0] if single else tuple(out)


# ## Calving internals
# 
# Mesh connectivity and iceberg detection for `position_law`. Ice counts as attached when it is
# connected through ice to grounded ice (vertices of fully grounded elements, as in ISSM's iceberg
# removal) or to the thickness-constrained inflow.

# In[ ]:


def _adjacency(md):

    st = _state(md)
    if getattr(st, 'adjacency', None) is None:
        import scipy.sparse as sp
        x, y, elts, nv2 = _mesh2d(md)
        i = elts[:, [0, 1, 2, 1, 2, 0]].ravel()
        j = elts[:, [1, 2, 0, 0, 1, 2]].ravel()
        st.adjacency = sp.coo_matrix((np.ones(i.size), (i, j)), shape = (nv2, nv2)).tocsr()
    return st.adjacency

def _components(A, nodes):

    from scipy.sparse.csgraph import connected_components
    labels = np.full(nodes.size, -1)
    idx = np.flatnonzero(nodes)
    if idx.size:
        labels[idx] = connected_components(A[idx][:, idx], directed = False)[1]
    return labels

def _cut_off(md, ice):

    #ice nodes not connected through ice to grounded ice (vertices of fully grounded elements, as in
    #ISSM's iceberg removal) or to the thickness-constrained (inflow) boundary
    x, y, elts, nv2 = _mesh2d(md)
    ocean = _to2d(md, md.mask.ocean_levelset)
    anchors = np.zeros(nv2, dtype = bool)
    anchors[elts[np.all(ocean[elts] > 0, axis = 1)].ravel()] = True
    anchors |= np.isfinite(_to2d(md, md.masstransport.spcthickness))
    anchors &= ice
    if not anchors.any():
        return np.zeros(nv2, dtype = bool)
    labels = _components(_adjacency(md), ice)
    return ice & ~np.isin(labels, labels[anchors])


# ## Ice fraction
# 
# The area of each element covered by ice, from the level set (linear on each element): used by
# `ice_volume`, and to measure what `position_law` calves.

# In[ ]:


def _element_areas(x, y, elts):

    xe, ye = x[elts], y[elts]
    return 0.5*np.abs((xe[:, 1] - xe[:, 0])*(ye[:, 2] - ye[:, 0]) - (xe[:, 2] - xe[:, 0])*(ye[:, 1] - ye[:, 0]))

def _ice_fraction(levelset, elts):

    φ = levelset[elts]
    neg = φ < 0
    n = neg.sum(axis = 1)
    frac = (n == 3).astype(float)
    rows = np.arange(len(elts))
    for count, lone in [(1, np.argmax(neg, axis = 1)), (2, np.argmin(neg, axis = 1))]: #the vertex alone on its side
        m = n == count
        a = φ[rows[m], lone[m]]
        others = φ[m][~np.eye(3, dtype = bool)[lone[m]]].reshape(-1, 2)
        corner = (a/(a - others[:, 0]))*(a/(a - others[:, 1])) #area fraction of the corner triangle at that vertex
        frac[m] = corner if count == 1 else 1 - corner
    return frac


# ## Front internals
# 
# For `diagnostic_solve` and `front_diagnostics`: the two level sets with the front on vertices
# (behind: the vertices of cut elements that are in the ice set to zero, so those elements are
# ice-free; ahead: their vertices outside the ice set to zero, so those elements are full of ice),
# and the interpolation weights, from every cut edge between an ice vertex i and an ice-free vertex j
# as -φ_i/(φ_j - φ_i), averaged on each vertex of the cut elements and extended to all others from
# the nearest of them. Integrals along the front use the zero contour of the level set in every
# element containing ice, with thickness and velocity interpolated linearly along it.

# In[ ]:


def _front_configurations(md):

    from scipy.spatial import cKDTree

    x, y, elts, nv2 = _mesh2d(md)
    φ = _to2d(md, md.mask.ice_levelset).astype(float)
    cut = np.any(φ[elts] < 0, axis = 1) & np.any(φ[elts] > 0, axis = 1)
    if not cut.any():
        return None
    in_cut = np.zeros(nv2, dtype = bool)
    in_cut[elts[cut].ravel()] = True
    behind, ahead = φ.copy(), φ.copy()
    behind[in_cut & (φ < 0)] = 0.
    ahead[in_cut & (φ > 0)] = 0.

    e = np.vstack([elts[cut][:, [0, 1]], elts[cut][:, [1, 2]], elts[cut][:, [2, 0]]])
    e = e[(φ[e[:, 0]] < 0) & (φ[e[:, 1]] > 0) | (φ[e[:, 0]] > 0) & (φ[e[:, 1]] < 0)]
    i = np.where(φ[e[:, 0]] < 0, e[:, 0], e[:, 1])
    j = np.where(φ[e[:, 0]] < 0, e[:, 1], e[:, 0])
    crossed = -φ[i]/(φ[j] - φ[i])
    count = np.bincount(np.r_[i, j], minlength = nv2)
    band = count > 0
    θ_band = np.bincount(np.r_[i, j], np.r_[crossed, crossed], nv2)[band]/count[band]
    θ = θ_band[cKDTree(np.c_[x[band], y[band]]).query(np.c_[x, y])[1]]
    return behind, ahead, θ

def _blend_outputs(md, sol_behind, sol_ahead, w):

    names = [o for o in md.stressbalance.requested_outputs if o != 'default']
    blended = SimpleNamespace()
    for name in names:
        if hasattr(sol_ahead, name):
            a = np.asarray(getattr(sol_ahead, name), dtype = float).ravel()
            b = np.asarray(getattr(sol_behind, name), dtype = float).ravel() if hasattr(sol_behind, name) else a
            setattr(blended, name, (1 - w)*b + w*a if a.size == w.size == b.size else a)
    return blended

def _depth_average(md, field):

    f = np.asarray(field, dtype = float).ravel()
    if not _state(md).is3d:
        return f
    nl = int(md.mesh.numberoflayers)
    f = f.reshape(nl, -1)
    return (f[1:] + f[:-1]).mean(axis = 0)/2 #trapezoidal rule on the (evenly spaced in σ) layers

def _contour_integrals(x, y, elts, φ, H, u):

    #length, flux (∫ H u·n ds) and ∫ u·n ds along the zero contour of φ, over elements containing ice
    length = flux = speed = 0.
    for el in elts[np.min(φ[elts], axis = 1) < 0]:
        f = φ[el]
        if not np.any(f >= 0):
            continue
        pts = [] #(position, H, u) where the contour crosses an edge or passes through a vertex
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            fa, fb = f[a], f[b]
            if fa == 0:
                pts.append(el[a])
            if (fa < 0 < fb) or (fb < 0 < fa):
                t = fa/(fa - fb)
                pts.append((el[a], el[b], t))
        pts = list(dict.fromkeys(p if isinstance(p, tuple) else (p, p, 0.) for p in pts))
        if len(pts) < 2:
            continue
        def at(p):
            i, j, t = p
            return (np.array([x[i] + t*(x[j] - x[i]), y[i] + t*(y[j] - y[i])]), H[i] + t*(H[j] - H[i]),
                    np.array([u[0][i] + t*(u[0][j] - u[0][i]), u[1][i] + t*(u[1][j] - u[1][i])]))
        (p1, H1, u1), (p2, H2, u2) = at(pts[0]), at(pts[1])
        ds = np.hypot(*(p2 - p1))
        if ds == 0:
            continue
        g = np.linalg.solve(np.c_[x[el], y[el], np.ones(3)], f)[:2] #∇φ in the element: points out of the ice
        n = g/np.hypot(*g)
        length += ds
        flux += ds*(H1*(u1 @ n) + H2*(u2 @ n))/2
        speed += ds*((u1 @ n) + (u2 @ n))/2
    return length, flux, speed


# ## Flowline internals
# 
# The pairs of nodes across the strip (1-based, on every layer), leaving out nodes whose velocity or
# thickness is constrained, which are equal on both edges already; and the first place along the
# flowline where a field goes from negative to positive.

# In[ ]:


def _flowline_pairs(md):

    x, y, elts, nv2 = _mesh2d(md)
    free = np.isnan(_to2d(md, md.stressbalance.spcvx)) & np.isnan(_to2d(md, md.masstransport.spcthickness))
    a, b = np.flatnonzero(y == y.min()), np.flatnonzero(y == y.max())
    a, b = a[np.argsort(x[a])], b[np.argsort(x[b])]
    keep = free[a] & free[b]
    layers = md.mesh.numberoflayers if hasattr(md.mesh, 'x2d') else 1
    return np.vstack([np.c_[a[keep], b[keep]] + 1 + k*nv2 for k in range(layers)])

def _crossing(md, field):

    x, f = profile(md, field)
    k = np.flatnonzero((f[:-1] < 0) & (f[1:] >= 0))
    if k.size == 0:
        return np.nan
    k = k[0]
    return x[k] - f[k]*(x[k + 1] - x[k])/(f[k + 1] - f[k])


# ## Calling ISSM
# 
# `solve_quiet` calls ISSM without any console output, so progress bars stay clean: through the ISSM
# worker when it is available, or through ISSM's usual `solve` otherwise, reusing one run directory
# instead of creating a new one on every call. ISSM's consistency check runs on the first solve of each
# type for a model (`check = True` forces it on every call, `check = False` skips it). Output is
# captured rather than discarded, so if ISSM fails the exception includes ISSM's own error message.

# In[ ]:


class quiet_output:
    """Capture all console output (Python and ISSM's C++ core) inside a `with` block, instead of
    printing it; `text()` returns what was captured."""

    def __init__(self, active = True):
        self.active = active

    def __enter__(self):
        if self.active:
            self.python = io.StringIO()
            self.native = tempfile.TemporaryFile()
            self.saved = (os.dup(1), os.dup(2), sys.stdout, sys.stderr)
            sys.stdout = sys.stderr = self.python
            os.dup2(self.native.fileno(), 1); os.dup2(self.native.fileno(), 2)
        return self

    def __exit__(self, *exc):
        if self.active:
            o1, o2, s1, s2 = self.saved
            os.dup2(o1, 1); os.dup2(o2, 2)
            os.close(o1); os.close(o2)
            sys.stdout = s1; sys.stderr = s2
            self.native.seek(0)
            self.captured = self.python.getvalue() + self.native.read().decode(errors = 'replace')
            self.native.close()
        return False

    def text(self):
        return self.captured if self.active else ''

_failed_np = {} #process counts the worker could not run on in this session, and what mpiexec printed

def _one_process(md): #carry on with md on one process
    md.cluster.np = 1
    if dict(md.toolkits.DefaultAnalysis) == dict(iluasmoptions('ksp_max_it', 1000, 'ksp_rtol', 1e-10)) and not hasattr(md.toolkits, 'StressbalanceAnalysis'):
        md.toolkits.addoptions('StressbalanceAnalysis', {'toolkit': 'petsc', 'mat_type': 'aij', 'ksp_type': 'preonly', 'pc_type': 'lu'}) #as initialize_model does on one process without MUMPS

def solve_quiet(md, solution_type, quiet = True, check = None):

    if md.cluster.np > 1 and md.cluster.np in _failed_np:
        _one_process(md)
    st = getattr(md.miscellaneous, 'helper', None)
    if check is None:
        check = st is None or solution_type not in st.checked
    worker = _get_worker(md)

    start = time.time()
    with quiet_output(quiet) as captured:
        try:
            if worker is None:
                options = ['runtimename', False]
                if not check:
                    options += ['checkconsistency', 'no']
                    md.private.solution = solution_type + 'Solution' #solve() only sets this when it runs the check
                out, error = solve(md, solution_type, *options), None
            else:
                out, error = _solve_with_worker(md, solution_type, check, worker, quiet), None
        except Exception as err:
            out, error = None, err
    console = captured.text()

    if isinstance(error, _WorkerStopped) and md.cluster.np > 1:
        _failed_np[md.cluster.np] = str(error)
        warnings.warn(f'issm_helper: ISSM stopped when run on {md.cluster.np} processes (mpiexec -np {md.cluster.np}), so this model and all later ones '
                      'carry on on one process, which is slower. To use several processes, run the session with at least that many cores, '
                      'with the mpiexec of the MPI that ISSM was built with; with one process (md.cluster.np = 1) this warning goes away. '
                      f'What it printed:\n{error}', stacklevel = 2)
        _one_process(md)
        return solve_quiet(md, solution_type, quiet, check)
    if out is not None and hasattr(out.results, solution_type + 'Solution'):
        if st is not None and check:
            st.checked.add(solution_type)
        return out
    raise RuntimeError(_failure_message(md, solution_type, error, console, start)) from error

def _excerpt(text):

    lines = [l for l in text.splitlines() if l.strip()]
    for key, length in [('???', 3), ('not consistent', 12), ('error', 12)]: #ISSM's C++ errors start with '???'
        flagged = [i for i, l in enumerate(lines) if key in l.lower()]
        if flagged:
            return '\n'.join(lines[flagged[0]:flagged[0] + length])
    return '\n'.join(lines[-20:])

def _failure_message(md, solution_type, error, console, start):

    message = f'ISSM {solution_type} solve failed: ' + (str(error) if error is not None else 'no results were produced.')

    name = md.miscellaneous.name + '.errlog'
    for path in [name, os.path.join(md.cluster.executionpath, md.private.runtimename, name),
                 os.path.join(md.cluster.executionpath, f'{md.private.runtimename}-{os.getpid()}', name)]:
        if os.path.exists(path) and os.path.getmtime(path) >= start - 1 and os.path.getsize(path) > 0: #written by this run
            with open(path) as f:
                message += '\n--- ISSM errlog ---\n' + f.read().strip()
            break

    shown = _excerpt(console)
    if shown:
        message += '\n--- ISSM output ---\n' + shown
    return message


# ## ISSM worker
# 
# ISSM normally starts a new program for every solve: it initializes MPI and PETSc, reads the model
# file, solves, writes the results and exits. On small meshes that start-up and shutdown take about as
# long as the time step itself. The worker is ISSM's own main program (`issm.cpp`) wrapped in a loop:
# it initializes once, then solves one model file per request until Python closes it. Every request
# still writes the whole model and ISSM builds it from scratch, so any change to `md` between steps is
# honored and the numerics are ISSM's own.
# 
# The worker is built automatically the first time it is needed (when a model is set up with
# `initialize_model`), and rebuilt after ISSM is reinstalled. `build_worker()` generates it from your
# ISSM's `issm.cpp`, compiles and links it with the exact commands `make` uses for `issm.exe`, and
# installs it next to `issm.exe`. It needs the directory where ISSM was compiled, which it looks for in
# `ISSM_DIR` and in the directories above the current one, and the compilers ISSM was built with (on a
# cluster, the same modules loaded). If it cannot be built, a warning says why and the solves use
# ISSM's usual `solve`; `build_worker(issm_dir = '/path/to/issm')` builds it by hand, and
# `USE_WORKER = False` switches it off.
# 
# On several processes (`md.cluster.np` > 1), the worker runs as `mpiexec -np md.cluster.np` on this
# machine. In a session on a cluster node, `mpiexec` may refuse to start more processes than the job
# has tasks, or try to start them through SLURM, so the worker lets Open MPI start more processes than
# tasks and has MPICH and Intel MPI start them directly on this node (the environment variables in
# `LOCAL_MPI`, unless they are already set). If the worker still stops on several processes, before or
# during a solve, `solve_quiet` warns, with what `mpiexec` printed, and carries on with that model and
# all later ones on one process.

# In[ ]:


USE_WORKER = True #False: always use ISSM's usual solve
LOCAL_MPI = {'OMPI_MCA_rmaps_base_oversubscribe': '1', #Open MPI: start more processes than the job's task count
             'HYDRA_LAUNCHER': 'fork', 'I_MPI_HYDRA_BOOTSTRAP': 'fork'} #MPICH, Intel MPI: start them on this node, not through srun

_WORKER_TEMPLATE = r"""/* issm_worker.cpp, generated by issm_helper.build_worker() from ISSM's own main program (issm.cpp):
 * the same steps, repeated for every request "<SolutionType> <directory> <model name>" read on standard
 * input, with MPI and PETSc initialized only once. Prints "ISSM_WORKER_DONE <status>" after each request
 * and stops at "quit" or at the end of the input. */
{includes}
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

int main(int main_argc, char **main_argv){{

	int argc = main_argc;
	char **argv = main_argv;
	{init}
	int rank = 0;
	ISSM_MPI_Comm_rank({comm}, &rank);

	char request[4096];
	for(;;){{

		/*read the request on rank 0 and share it with the other ranks*/
		int length = 0;
		if(rank == 0){{
			std::string line;
			if(!std::getline(std::cin, line)) line = "quit";
			strncpy(request, line.c_str(), sizeof(request) - 1);
			request[sizeof(request) - 1] = '\0';
			length = (int)strlen(request);
		}}
		ISSM_MPI_Bcast(&length, 1, ISSM_MPI_INT, 0, {comm});
		ISSM_MPI_Bcast(request, length + 1, ISSM_MPI_CHAR, 0, {comm});
		if(length == 0 || strcmp(request, "quit") == 0) break;

		std::istringstream words(request);
		std::string solution, directory, name;
		words >> solution >> directory >> name;
		char *arguments[4] = {{main_argv[0], (char*)solution.c_str(), (char*)directory.c_str(), (char*)name.c_str()}};

		/*the body of ISSM's main program*/
		int status = 0;
		try{{
			int argc = 4;
			char **argv = arguments;
{body}
		}}
		catch(ErrorException &e){{ e.Report(); status = 1; }}
		catch(std::exception &e){{ _printf_("Standard exception: " << e.what() << "\n"); status = 1; }}

		if(rank == 0){{
			fflush(stdout);
			printf("ISSM_WORKER_DONE %i\n", status);
			fflush(stdout);
		}}
	}}

	EnvironmentFinalize();
	return 0;
}}
"""

def build_worker(**kwargs):

    #the directory ISSM was compiled in (it has src/c/Makefile): given, or ISSM_DIR, or above the current directory
    candidates = [kwargs['issm_dir']] if kwargs.get('issm_dir') else [os.environ.get('ISSM_DIR')]
    if not kwargs.get('issm_dir'):
        path = os.path.abspath(os.getcwd())
        while True:
            candidates.append(path)
            if os.path.dirname(path) == path:
                break
            path = os.path.dirname(path)
    build = next((c for c in candidates if c and os.path.isfile(os.path.join(c, 'src', 'c', 'Makefile'))), None)
    if build is None:
        raise RuntimeError('build_worker needs the directory where ISSM was compiled (the one containing src/c/Makefile), '
                           f'and found none in: {", ".join(c for c in candidates if c)}. If ISSM was installed somewhere '
                           "else than where it was built, pass that directory: build_worker(issm_dir = '/path/to/issm').")
    src = os.path.join(os.path.abspath(build), 'src', 'c')

    #the worker: ISSM's own main program, run in a loop (for out-of-tree builds, the sources are in srcdir)
    main_cpp = os.path.join(src, 'main', 'issm.cpp')
    if not os.path.isfile(main_cpp):
        with open(os.path.join(src, 'Makefile')) as f:
            srcdir = re.search(r'^srcdir\s*=\s*(.*)$', f.read(), re.M).group(1).strip()
        main_cpp = os.path.join(src, srcdir, 'main', 'issm.cpp')
    if not os.path.isfile(main_cpp):
        raise RuntimeError(f'build_worker: found ISSM\'s build files in {src} but not its main program (main/issm.cpp).')
    with open(main_cpp) as f:
        main = f.read().split('\n')
    first = next(i for i, l in enumerate(main) if 'EnvironmentInit' in l)
    last = next(i for i, l in enumerate(main) if 'EnvironmentFinalize' in l)
    source = _WORKER_TEMPLATE.format(includes = '\n'.join(l for l in main if l.strip().startswith('#include')),
                                     init = main[first].strip(),
                                     comm = re.search(r'(\w+)\s*=\s*EnvironmentInit', main[first]).group(1),
                                     body = '\n'.join('\t\t' + l for l in main[first + 1:last]))
    os.makedirs(os.path.join(src, 'main'), exist_ok = True)
    with open(os.path.join(src, 'main', 'issm_worker.cpp'), 'w') as f:
        f.write(source)

    #compile and link it with the commands make uses for issm.exe
    dry = subprocess.run(['make', '-n', 'V=1', '-W', 'main/issm.cpp', 'issm.exe'], cwd = src, capture_output = True, text = True)
    commands = [c for c in dry.stdout.replace('\\\n', ' ').splitlines() if 'main/issm.cpp' in c or '-o issm.exe' in c]
    if dry.returncode != 0 or len(commands) < 2:
        raise RuntimeError(f'build_worker: could not find the commands that build issm.exe in {src}\n{dry.stdout}{dry.stderr}')
    commands = [re.sub(r'main/(issm-)?issm\.(o|obj)\b', r'main/issm_worker.\2',
                       c.replace('main/issm.cpp', 'main/issm_worker.cpp').replace('-o issm.exe', '-o issm_worker.exe')) for c in commands]

    #install it next to issm.exe (libtool relinks it for its installed location, as make install does)
    from generic import generic
    bindir = kwargs.get('bindir', generic().codepath)
    libtool = next((w for w in commands[-1].split() if w.endswith('libtool')), None)
    install = f'/bin/bash {libtool} --mode=install install -c issm_worker.exe {bindir}/issm_worker.exe' if libtool \
              else f'cp issm_worker.exe {bindir}/issm_worker.exe'

    for command in commands + [install]:
        r = subprocess.run(command, shell = True, cwd = src, capture_output = True, text = True, executable = '/bin/bash')
        if r.returncode != 0:
            raise RuntimeError(f'build_worker failed at:\n{command}\n{r.stdout}{r.stderr}')
    _close_worker()
    print(f'ISSM worker installed: {bindir}/issm_worker.exe')

class _Worker:

    def __init__(self, command):
        self.command = command
        env = {**{k: v for k, v in LOCAL_MPI.items() if k not in os.environ}, **os.environ}
        self.process = subprocess.Popen(command, stdin = subprocess.PIPE, stdout = subprocess.PIPE,
                                        stderr = subprocess.STDOUT, text = True, bufsize = 1, env = env)

    def run(self, solution, directory, name, echo = False):
        log = []
        try:
            self.process.stdin.write(f'{solution} {directory} {name}\n')
            self.process.stdin.flush()
        except OSError:
            return -1, self._remains(log) #the worker had already stopped, or never started
        while True:
            line = self.process.stdout.readline()
            if not line:
                return -1, self._remains(log) #the worker has stopped
            if line.startswith('ISSM_WORKER_DONE'):
                return int(line.split()[1]), ''.join(log)
            log.append(line)
            if echo:
                print(line, end = '')

    def _remains(self, log): #everything the stopped worker (or mpiexec) printed, and its exit code
        try:
            rest = self.process.communicate(timeout = 10)[0] or ''
        except Exception:
            self.process.kill()
            rest = ''
        return ''.join(log) + rest + f'(mpiexec -np {self.command[2]} exited with code {self.process.poll()})\n'

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('quit\n')
                self.process.stdin.flush()
                self.process.wait(timeout = 10)
            except Exception:
                self.process.kill()

class _WorkerStopped(RuntimeError): #the worker's processes ended, rather than ISSM reporting a failure
    pass

_worker = None
_worker_unavailable = False #set when building the worker failed in this session

def _get_worker(md):

    global _worker, _worker_unavailable
    codepath = getattr(md.cluster, 'codepath', None)
    if not USE_WORKER or _worker_unavailable or codepath is None:
        return None

    #build it the first time it is needed, and again after ISSM has been reinstalled
    exe, issm = os.path.join(codepath, 'issm_worker.exe'), os.path.join(codepath, 'issm.exe')
    if not os.path.isfile(exe) or (os.path.isfile(issm) and os.path.getmtime(issm) > os.path.getmtime(exe)):
        print('issm_helper: building the ISSM worker for this ISSM installation (only needed once)...', flush = True)
        try:
            with quiet_output():
                build_worker()
        except Exception as err:
            _worker_unavailable = True
            warnings.warn(f"issm_helper: could not build the ISSM worker ({' '.join(str(err).split())[:300]}). "
                          "Solves will use ISSM's usual solve, which is slower. To build it by hand: "
                          "build_worker(issm_dir = '/path/to/issm'); to stop trying: USE_WORKER = False.")
            return None
        print('issm_helper: ISSM worker ready.', flush = True)

    command = ['mpiexec', '-np', str(md.cluster.np), exe]
    if _worker is None or _worker.process.poll() is not None or _worker.command != command:
        _close_worker()
        _worker = _Worker(command)
    return _worker

def _close_worker():

    global _worker
    if _worker is not None:
        _worker.close()
        _worker = None

atexit.register(_close_worker)

def _solve_with_worker(md, solution_type, check, worker, quiet):

    solution = solution_type + 'Solution'
    md.private.solution = solution
    md.private.runtimename = md.miscellaneous.name
    if check:
        ismodelselfconsistent(md)

    root = os.path.join(md.cluster.executionpath, f'{md.private.runtimename}-{os.getpid()}') #one directory per process, so several can run at once
    os.makedirs(root, exist_ok = True)
    base = os.path.join(root, md.miscellaneous.name)
    if os.path.exists(base + '.outbin'):
        os.remove(base + '.outbin')
    if len(inspect.signature(marshall).parameters) > 1: #recent ISSM: marshall(md, filename)
        marshall(md, base + '.bin')
    else: #older ISSM: marshall(md) writes <name>.bin in the current directory
        cwd = os.getcwd()
        os.chdir(root)
        try:
            marshall(md)
        finally:
            os.chdir(cwd)
    md.toolkits.ToolkitsFile(base + '.toolkits')

    status, log = worker.run(solution, root, md.miscellaneous.name, echo = not quiet)
    if status != 0 or not os.path.exists(base + '.outbin'):
        _close_worker() #start a fresh worker next time
        raise (_WorkerStopped if status == -1 else RuntimeError)(_excerpt(log) or 'the ISSM worker stopped unexpectedly.')
    return loadresultsfromdisk(md, base + '.outbin')


# In[ ]:


# !jupyter nbconvert --to script issm_helper.ipynb

