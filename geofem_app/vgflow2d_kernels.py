"""Numba kernels for the VGFlow 2D public seepage substitute."""

from __future__ import annotations

import math

import numpy as np

from .fem2d_elements import _quad4_shape_grad_numba, _quad8_shape_grad_numba
from .fem2d_types import njit


VGFLOW_MATERIAL_SATURATED = 0
VGFLOW_MATERIAL_VAN_GENUCHTEN = 1
VGFLOW_MATERIAL_TABLE = 2


@njit(cache=True)
def _vgflow_tri3_shape_grad_numba(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    N = np.empty(3, dtype=np.float64)
    N[0] = 1.0 / 3.0
    N[1] = 1.0 / 3.0
    N[2] = 1.0 / 3.0
    dxi = np.empty(3, dtype=np.float64)
    deta = np.empty(3, dtype=np.float64)
    dxi[0] = -1.0
    dxi[1] = 1.0
    dxi[2] = 0.0
    deta[0] = -1.0
    deta[1] = 0.0
    deta[2] = 1.0
    j00 = 0.0
    j01 = 0.0
    j10 = 0.0
    j11 = 0.0
    for i in range(3):
        x = coords[i, 0]
        y = coords[i, 1]
        j00 += dxi[i] * x
        j01 += dxi[i] * y
        j10 += deta[i] * x
        j11 += deta[i] * y
    det = j00 * j11 - j01 * j10
    grad = np.zeros((2, 3), dtype=np.float64)
    if det <= 0.0:
        return N, grad, det
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    for i in range(3):
        grad[0, i] = inv00 * dxi[i] + inv01 * deta[i]
        grad[1, i] = inv10 * dxi[i] + inv11 * deta[i]
    return N, grad, det


@njit(cache=True)
def _vgflow_tri3_shape_grad_at_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    N = np.empty(3, dtype=np.float64)
    N[0] = 1.0 - xi - eta
    N[1] = xi
    N[2] = eta
    _N_center, grad, det = _vgflow_tri3_shape_grad_numba(coords)
    return N, grad, det


@njit(cache=True)
def _vgflow_tri6_shape_grad_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    l1 = 1.0 - xi - eta
    l2 = xi
    l3 = eta
    N = np.empty(6, dtype=np.float64)
    N[0] = l1 * (2.0 * l1 - 1.0)
    N[1] = l2 * (2.0 * l2 - 1.0)
    N[2] = l3 * (2.0 * l3 - 1.0)
    N[3] = 4.0 * l1 * l2
    N[4] = 4.0 * l2 * l3
    N[5] = 4.0 * l3 * l1

    dxi = np.empty(6, dtype=np.float64)
    deta = np.empty(6, dtype=np.float64)
    dxi[0] = -(4.0 * l1 - 1.0)
    dxi[1] = 4.0 * l2 - 1.0
    dxi[2] = 0.0
    dxi[3] = 4.0 * (l1 - l2)
    dxi[4] = 4.0 * l3
    dxi[5] = -4.0 * l3
    deta[0] = -(4.0 * l1 - 1.0)
    deta[1] = 0.0
    deta[2] = 4.0 * l3 - 1.0
    deta[3] = -4.0 * l2
    deta[4] = 4.0 * l2
    deta[5] = 4.0 * (l1 - l3)

    j00 = 0.0
    j01 = 0.0
    j10 = 0.0
    j11 = 0.0
    for i in range(6):
        x = coords[i, 0]
        y = coords[i, 1]
        j00 += dxi[i] * x
        j01 += dxi[i] * y
        j10 += deta[i] * x
        j11 += deta[i] * y
    det = j00 * j11 - j01 * j10
    grad = np.zeros((2, 6), dtype=np.float64)
    if det <= 0.0:
        return N, grad, det
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    for i in range(6):
        grad[0, i] = inv00 * dxi[i] + inv01 * deta[i]
        grad[1, i] = inv10 * dxi[i] + inv11 * deta[i]
    return N, grad, det


@njit(cache=True)
def _vgflow_shape_grad_at_numba(coords: np.ndarray, element_type_code: int, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    if element_type_code == 3:
        return _vgflow_tri3_shape_grad_at_numba(coords, xi, eta)
    if element_type_code == 4:
        return _quad4_shape_grad_numba(coords, xi, eta)
    if element_type_code == 6:
        return _vgflow_tri6_shape_grad_numba(coords, xi, eta)
    return _quad8_shape_grad_numba(coords, xi, eta)


@njit(cache=True)
def vgflow_water_state_numba(
    model_code: int,
    pressure_head: float,
    alpha: float,
    n: float,
    theta_r: float,
    theta_s: float,
) -> tuple[float, float, float, float]:
    if model_code == VGFLOW_MATERIAL_VAN_GENUCHTEN:
        if pressure_head >= 0.0:
            return theta_s, 1.0, 1.0, 0.0
        suction = -pressure_head
        m = 1.0 - 1.0 / n
        alpha_s = alpha * suction
        term = 1.0 + alpha_s**n
        se = term ** (-m)
        theta = theta_r + se * (theta_s - theta_r)
        inner = 1.0 - (1.0 - se ** (1.0 / m)) ** m
        if inner < 0.0:
            inner = 0.0
        kr = math.sqrt(max(se, 0.0)) * inner * inner
        if kr < 0.0:
            kr = 0.0
        elif kr > 1.0:
            kr = 1.0
        dse_dpsi = m * n * alpha**n * suction ** (n - 1.0) * term ** (-m - 1.0)
        capacity = (theta_s - theta_r) * dse_dpsi
        if capacity < 0.0:
            capacity = 0.0
        return theta, se, kr, capacity
    return theta_s, 1.0, 1.0, 0.0


@njit(cache=True)
def vgflow_water_state_array_numba(
    model_code: int,
    pressure_heads: np.ndarray,
    alpha: float,
    n: float,
    theta_r: float,
    theta_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = pressure_heads.shape[0]
    theta = np.empty(count, dtype=np.float64)
    saturation = np.empty(count, dtype=np.float64)
    kr = np.empty(count, dtype=np.float64)
    capacity = np.empty(count, dtype=np.float64)
    for i in range(count):
        theta_i, sat_i, kr_i, cap_i = vgflow_water_state_numba(model_code, pressure_heads[i], alpha, n, theta_r, theta_s)
        theta[i] = theta_i
        saturation[i] = sat_i
        kr[i] = kr_i
        capacity[i] = cap_i
    return theta, saturation, kr, capacity


@njit(cache=True)
def vgflow_table_state_numba(
    table_values: np.ndarray,
    start: int,
    stop: int,
    pressure_head: float,
    theta_r: float,
    theta_s: float,
) -> tuple[float, float, float, float, float]:
    count = stop - start
    if count <= 0:
        return pressure_head, theta_s, 1.0, 1.0, 0.0

    first_psi = table_values[start, 0]
    last_psi = table_values[stop - 1, 0]
    if pressure_head <= first_psi:
        psi = first_psi
        theta = table_values[start, 1]
        kr = table_values[start, 2]
    elif pressure_head >= last_psi:
        psi = last_psi
        theta = table_values[stop - 1, 1]
        kr = table_values[stop - 1, 2]
    else:
        psi = pressure_head
        theta = table_values[stop - 1, 1]
        kr = table_values[stop - 1, 2]
        for i in range(start, stop - 1):
            left_psi = table_values[i, 0]
            right_psi = table_values[i + 1, 0]
            if left_psi <= pressure_head <= right_psi:
                denom = right_psi - left_psi
                if abs(denom) < np.finfo(np.float64).eps:
                    denom = np.finfo(np.float64).eps
                t = (pressure_head - left_psi) / denom
                theta = table_values[i, 1] + t * (table_values[i + 1, 1] - table_values[i, 1])
                kr = table_values[i, 2] + t * (table_values[i + 1, 2] - table_values[i, 2])
                break

    if kr < 0.0:
        kr = 0.0
    elif kr > 1.0:
        kr = 1.0

    capacity = 0.0
    if count >= 2:
        nearest = start
        best = abs(0.5 * (table_values[start, 0] + table_values[start + 1, 0]) - pressure_head)
        for i in range(start + 1, stop - 1):
            distance = abs(0.5 * (table_values[i, 0] + table_values[i + 1, 0]) - pressure_head)
            if distance < best:
                best = distance
                nearest = i
        denom = table_values[nearest + 1, 0] - table_values[nearest, 0]
        if abs(denom) < np.finfo(np.float64).eps:
            denom = np.finfo(np.float64).eps
        capacity = (table_values[nearest + 1, 1] - table_values[nearest, 1]) / denom
        if capacity < 0.0:
            capacity = 0.0

    denom_sat = theta_s - theta_r
    if denom_sat < np.finfo(np.float64).eps:
        denom_sat = np.finfo(np.float64).eps
    saturation = (theta - theta_r) / denom_sat
    if saturation < 0.0:
        saturation = 0.0
    elif saturation > 1.0:
        saturation = 1.0
    return psi, theta, saturation, kr, capacity


@njit(cache=True)
def vgflow_table_state_array_numba(
    table_values: np.ndarray,
    start: int,
    stop: int,
    pressure_heads: np.ndarray,
    theta_r: float,
    theta_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = pressure_heads.shape[0]
    psi = np.empty(count, dtype=np.float64)
    theta = np.empty(count, dtype=np.float64)
    saturation = np.empty(count, dtype=np.float64)
    kr = np.empty(count, dtype=np.float64)
    capacity = np.empty(count, dtype=np.float64)
    for i in range(count):
        psi_i, theta_i, sat_i, kr_i, cap_i = vgflow_table_state_numba(table_values, start, stop, pressure_heads[i], theta_r, theta_s)
        psi[i] = psi_i
        theta[i] = theta_i
        saturation[i] = sat_i
        kr[i] = kr_i
        capacity[i] = cap_i
    return psi, theta, saturation, kr, capacity


@njit(cache=True)
def vgflow_quad4_assembly_triplets_numba(
    coords: np.ndarray,
    connectivity: np.ndarray,
    material_ids: np.ndarray,
    head: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    specific_storage: np.ndarray,
    angle_rad: np.ndarray,
    model_codes: np.ndarray,
    table_offsets: np.ndarray,
    table_values: np.ndarray,
    alpha: np.ndarray,
    n_values: np.ndarray,
    theta_r: np.ndarray,
    theta_s: np.ndarray,
    horizontal_pressure: bool,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    element_count = connectivity.shape[0]
    entry_count = element_count * 16
    rows = np.empty(entry_count, dtype=np.int64)
    cols = np.empty(entry_count, dtype=np.int64)
    k_data = np.empty(entry_count, dtype=np.float64)
    m_data = np.empty(entry_count, dtype=np.float64)
    local_coords = np.empty((4, 2), dtype=np.float64)
    a = 1.0 / math.sqrt(3.0)
    out = 0

    for elem in range(element_count):
        mat = material_ids[elem]
        avg_pressure = 0.0
        for i in range(4):
            node = connectivity[elem, i]
            x = coords[node, 0]
            y = coords[node, 1]
            local_coords[i, 0] = x
            local_coords[i, 1] = y
            if horizontal_pressure:
                avg_pressure += head[node]
            else:
                avg_pressure += head[node] - y
        avg_pressure *= 0.25
        if model_codes[mat] == VGFLOW_MATERIAL_TABLE:
            _psi, _theta, _sat, kr, capacity = vgflow_table_state_numba(
                table_values,
                table_offsets[mat],
                table_offsets[mat + 1],
                avg_pressure,
                theta_r[mat],
                theta_s[mat],
            )
        else:
            _theta, _sat, kr, capacity = vgflow_water_state_numba(
                model_codes[mat],
                avg_pressure,
                alpha[mat],
                n_values[mat],
                theta_r[mat],
                theta_s[mat],
            )

        c = math.cos(angle_rad[mat])
        s = math.sin(angle_rad[mat])
        kxx = (kx[mat] * c * c + ky[mat] * s * s) * kr
        kxy = ((kx[mat] - ky[mat]) * c * s) * kr
        kyy = (kx[mat] * s * s + ky[mat] * c * c) * kr
        storage = specific_storage[mat] + capacity

        ke = np.zeros((4, 4), dtype=np.float64)
        me = np.zeros((4, 4), dtype=np.float64)
        for gp in range(4):
            xi = -a if gp == 0 or gp == 3 else a
            eta = -a if gp == 0 or gp == 1 else a
            N, grad, det = _quad4_shape_grad_numba(local_coords, xi, eta)
            if det <= 0.0:
                return rows, cols, k_data, m_data, elem, det
            scale = det
            if axisymmetric:
                radius = 0.0
                for i in range(4):
                    radius += N[i] * local_coords[i, 0]
                if radius < np.finfo(np.float64).eps:
                    radius = np.finfo(np.float64).eps
                scale *= 2.0 * math.pi * radius
            for i in range(4):
                gix = grad[0, i]
                giy = grad[1, i]
                for j in range(4):
                    gjx = grad[0, j]
                    gjy = grad[1, j]
                    ke[i, j] += (gix * kxx * gjx + gix * kxy * gjy + giy * kxy * gjx + giy * kyy * gjy) * scale
                    me[i, j] += storage * N[i] * N[j] * scale

        for i in range(4):
            row = connectivity[elem, i]
            for j in range(4):
                rows[out] = row
                cols[out] = connectivity[elem, j]
                k_data[out] = ke[i, j]
                m_data[out] = me[i, j]
                out += 1
    return rows, cols, k_data, m_data, -1, 0.0


@njit(cache=True)
def vgflow_quad8_assembly_triplets_numba(
    coords: np.ndarray,
    connectivity: np.ndarray,
    material_ids: np.ndarray,
    head: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    specific_storage: np.ndarray,
    angle_rad: np.ndarray,
    model_codes: np.ndarray,
    table_offsets: np.ndarray,
    table_values: np.ndarray,
    alpha: np.ndarray,
    n_values: np.ndarray,
    theta_r: np.ndarray,
    theta_s: np.ndarray,
    horizontal_pressure: bool,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    element_count = connectivity.shape[0]
    entry_count = element_count * 64
    rows = np.empty(entry_count, dtype=np.int64)
    cols = np.empty(entry_count, dtype=np.int64)
    k_data = np.empty(entry_count, dtype=np.float64)
    m_data = np.empty(entry_count, dtype=np.float64)
    local_coords = np.empty((8, 2), dtype=np.float64)
    gp_xi = np.empty(3, dtype=np.float64)
    gp_w = np.empty(3, dtype=np.float64)
    a = math.sqrt(3.0 / 5.0)
    gp_xi[0] = -a
    gp_xi[1] = 0.0
    gp_xi[2] = a
    gp_w[0] = 5.0 / 9.0
    gp_w[1] = 8.0 / 9.0
    gp_w[2] = 5.0 / 9.0
    out = 0

    for elem in range(element_count):
        mat = material_ids[elem]
        avg_pressure = 0.0
        for i in range(8):
            node = connectivity[elem, i]
            x = coords[node, 0]
            y = coords[node, 1]
            local_coords[i, 0] = x
            local_coords[i, 1] = y
            if horizontal_pressure:
                avg_pressure += head[node]
            else:
                avg_pressure += head[node] - y
        avg_pressure *= 0.125
        if model_codes[mat] == VGFLOW_MATERIAL_TABLE:
            _psi, _theta, _sat, kr, capacity = vgflow_table_state_numba(
                table_values,
                table_offsets[mat],
                table_offsets[mat + 1],
                avg_pressure,
                theta_r[mat],
                theta_s[mat],
            )
        else:
            _theta, _sat, kr, capacity = vgflow_water_state_numba(
                model_codes[mat],
                avg_pressure,
                alpha[mat],
                n_values[mat],
                theta_r[mat],
                theta_s[mat],
            )

        c = math.cos(angle_rad[mat])
        s = math.sin(angle_rad[mat])
        kxx = (kx[mat] * c * c + ky[mat] * s * s) * kr
        kxy = ((kx[mat] - ky[mat]) * c * s) * kr
        kyy = (kx[mat] * s * s + ky[mat] * c * c) * kr
        storage = specific_storage[mat] + capacity

        ke = np.zeros((8, 8), dtype=np.float64)
        me = np.zeros((8, 8), dtype=np.float64)
        for gx in range(3):
            xi = gp_xi[gx]
            wx = gp_w[gx]
            for gy in range(3):
                eta = gp_xi[gy]
                weight = wx * gp_w[gy]
                N, grad, det = _quad8_shape_grad_numba(local_coords, xi, eta)
                if det <= 0.0:
                    return rows, cols, k_data, m_data, elem, det
                scale = det * weight
                if axisymmetric:
                    radius = 0.0
                    for i in range(8):
                        radius += N[i] * local_coords[i, 0]
                    if radius < np.finfo(np.float64).eps:
                        radius = np.finfo(np.float64).eps
                    scale *= 2.0 * math.pi * radius
                for i in range(8):
                    gix = grad[0, i]
                    giy = grad[1, i]
                    for j in range(8):
                        gjx = grad[0, j]
                        gjy = grad[1, j]
                        ke[i, j] += (gix * kxx * gjx + gix * kxy * gjy + giy * kxy * gjx + giy * kyy * gjy) * scale
                        me[i, j] += storage * N[i] * N[j] * scale

        for i in range(8):
            row = connectivity[elem, i]
            for j in range(8):
                rows[out] = row
                cols[out] = connectivity[elem, j]
                k_data[out] = ke[i, j]
                m_data[out] = me[i, j]
                out += 1
    return rows, cols, k_data, m_data, -1, 0.0


@njit(cache=True)
def vgflow_tri3_assembly_triplets_numba(
    coords: np.ndarray,
    connectivity: np.ndarray,
    material_ids: np.ndarray,
    head: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    specific_storage: np.ndarray,
    angle_rad: np.ndarray,
    model_codes: np.ndarray,
    table_offsets: np.ndarray,
    table_values: np.ndarray,
    alpha: np.ndarray,
    n_values: np.ndarray,
    theta_r: np.ndarray,
    theta_s: np.ndarray,
    horizontal_pressure: bool,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    element_count = connectivity.shape[0]
    entry_count = element_count * 9
    rows = np.empty(entry_count, dtype=np.int64)
    cols = np.empty(entry_count, dtype=np.int64)
    k_data = np.empty(entry_count, dtype=np.float64)
    m_data = np.empty(entry_count, dtype=np.float64)
    local_coords = np.empty((3, 2), dtype=np.float64)
    out = 0

    for elem in range(element_count):
        mat = material_ids[elem]
        avg_pressure = 0.0
        for i in range(3):
            node = connectivity[elem, i]
            x = coords[node, 0]
            y = coords[node, 1]
            local_coords[i, 0] = x
            local_coords[i, 1] = y
            if horizontal_pressure:
                avg_pressure += head[node]
            else:
                avg_pressure += head[node] - y
        avg_pressure /= 3.0
        if model_codes[mat] == VGFLOW_MATERIAL_TABLE:
            _psi, _theta, _sat, kr, capacity = vgflow_table_state_numba(
                table_values,
                table_offsets[mat],
                table_offsets[mat + 1],
                avg_pressure,
                theta_r[mat],
                theta_s[mat],
            )
        else:
            _theta, _sat, kr, capacity = vgflow_water_state_numba(
                model_codes[mat],
                avg_pressure,
                alpha[mat],
                n_values[mat],
                theta_r[mat],
                theta_s[mat],
            )

        c = math.cos(angle_rad[mat])
        s = math.sin(angle_rad[mat])
        kxx = (kx[mat] * c * c + ky[mat] * s * s) * kr
        kxy = ((kx[mat] - ky[mat]) * c * s) * kr
        kyy = (kx[mat] * s * s + ky[mat] * c * c) * kr
        storage = specific_storage[mat] + capacity

        N, grad, det = _vgflow_tri3_shape_grad_numba(local_coords)
        if det <= 0.0:
            return rows, cols, k_data, m_data, elem, det
        scale = det * 0.5
        if axisymmetric:
            radius = 0.0
            for i in range(3):
                radius += N[i] * local_coords[i, 0]
            if radius < np.finfo(np.float64).eps:
                radius = np.finfo(np.float64).eps
            scale *= 2.0 * math.pi * radius
        for i in range(3):
            row = connectivity[elem, i]
            gix = grad[0, i]
            giy = grad[1, i]
            for j in range(3):
                gjx = grad[0, j]
                gjy = grad[1, j]
                rows[out] = row
                cols[out] = connectivity[elem, j]
                k_data[out] = (gix * kxx * gjx + gix * kxy * gjy + giy * kxy * gjx + giy * kyy * gjy) * scale
                m_data[out] = storage * N[i] * N[j] * scale
                out += 1
    return rows, cols, k_data, m_data, -1, 0.0


@njit(cache=True)
def vgflow_tri6_assembly_triplets_numba(
    coords: np.ndarray,
    connectivity: np.ndarray,
    material_ids: np.ndarray,
    head: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    specific_storage: np.ndarray,
    angle_rad: np.ndarray,
    model_codes: np.ndarray,
    table_offsets: np.ndarray,
    table_values: np.ndarray,
    alpha: np.ndarray,
    n_values: np.ndarray,
    theta_r: np.ndarray,
    theta_s: np.ndarray,
    horizontal_pressure: bool,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    element_count = connectivity.shape[0]
    entry_count = element_count * 36
    rows = np.empty(entry_count, dtype=np.int64)
    cols = np.empty(entry_count, dtype=np.int64)
    k_data = np.empty(entry_count, dtype=np.float64)
    m_data = np.empty(entry_count, dtype=np.float64)
    local_coords = np.empty((6, 2), dtype=np.float64)
    gp_xi = np.empty(3, dtype=np.float64)
    gp_eta = np.empty(3, dtype=np.float64)
    gp_xi[0] = 1.0 / 6.0
    gp_eta[0] = 1.0 / 6.0
    gp_xi[1] = 2.0 / 3.0
    gp_eta[1] = 1.0 / 6.0
    gp_xi[2] = 1.0 / 6.0
    gp_eta[2] = 2.0 / 3.0
    out = 0

    for elem in range(element_count):
        mat = material_ids[elem]
        avg_pressure = 0.0
        for i in range(6):
            node = connectivity[elem, i]
            x = coords[node, 0]
            y = coords[node, 1]
            local_coords[i, 0] = x
            local_coords[i, 1] = y
            if horizontal_pressure:
                avg_pressure += head[node]
            else:
                avg_pressure += head[node] - y
        avg_pressure /= 6.0
        if model_codes[mat] == VGFLOW_MATERIAL_TABLE:
            _psi, _theta, _sat, kr, capacity = vgflow_table_state_numba(
                table_values,
                table_offsets[mat],
                table_offsets[mat + 1],
                avg_pressure,
                theta_r[mat],
                theta_s[mat],
            )
        else:
            _theta, _sat, kr, capacity = vgflow_water_state_numba(
                model_codes[mat],
                avg_pressure,
                alpha[mat],
                n_values[mat],
                theta_r[mat],
                theta_s[mat],
            )

        c = math.cos(angle_rad[mat])
        s = math.sin(angle_rad[mat])
        kxx = (kx[mat] * c * c + ky[mat] * s * s) * kr
        kxy = ((kx[mat] - ky[mat]) * c * s) * kr
        kyy = (kx[mat] * s * s + ky[mat] * c * c) * kr
        storage = specific_storage[mat] + capacity

        ke = np.zeros((6, 6), dtype=np.float64)
        me = np.zeros((6, 6), dtype=np.float64)
        for gp in range(3):
            N, grad, det = _vgflow_tri6_shape_grad_numba(local_coords, gp_xi[gp], gp_eta[gp])
            if det <= 0.0:
                return rows, cols, k_data, m_data, elem, det
            scale = det / 6.0
            if axisymmetric:
                radius = 0.0
                for i in range(6):
                    radius += N[i] * local_coords[i, 0]
                if radius < np.finfo(np.float64).eps:
                    radius = np.finfo(np.float64).eps
                scale *= 2.0 * math.pi * radius
            for i in range(6):
                gix = grad[0, i]
                giy = grad[1, i]
                for j in range(6):
                    gjx = grad[0, j]
                    gjy = grad[1, j]
                    ke[i, j] += (gix * kxx * gjx + gix * kxy * gjy + giy * kxy * gjx + giy * kyy * gjy) * scale
                    me[i, j] += storage * N[i] * N[j] * scale

        for i in range(6):
            row = connectivity[elem, i]
            for j in range(6):
                rows[out] = row
                cols[out] = connectivity[elem, j]
                k_data[out] = ke[i, j]
                m_data[out] = me[i, j]
                out += 1
    return rows, cols, k_data, m_data, -1, 0.0


@njit(cache=True)
def vgflow_post_element_fields_numba(
    coords: np.ndarray,
    connectivity: np.ndarray,
    material_ids: np.ndarray,
    head: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    angle_rad: np.ndarray,
    model_codes: np.ndarray,
    table_offsets: np.ndarray,
    table_values: np.ndarray,
    alpha: np.ndarray,
    n_values: np.ndarray,
    theta_r: np.ndarray,
    theta_s: np.ndarray,
    horizontal_pressure: bool,
    element_type_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    element_count = connectivity.shape[0]
    node_count = connectivity.shape[1]
    centers = np.empty((element_count, 2), dtype=np.float64)
    gradients = np.empty((element_count, 2), dtype=np.float64)
    velocities = np.empty((element_count, 2), dtype=np.float64)
    bboxes = np.empty((element_count, 4), dtype=np.float64)
    local_coords = np.empty((node_count, 2), dtype=np.float64)

    for elem in range(element_count):
        mat = material_ids[elem]
        avg_head = 0.0
        min_x = 1.0e300
        min_y = 1.0e300
        max_x = -1.0e300
        max_y = -1.0e300
        for i in range(node_count):
            node = connectivity[elem, i]
            x = coords[node, 0]
            y = coords[node, 1]
            local_coords[i, 0] = x
            local_coords[i, 1] = y
            avg_head += head[node]
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        avg_head /= node_count

        N, grad, det = _vgflow_shape_grad_at_numba(local_coords, element_type_code, 0.0, 0.0)
        if det <= 0.0:
            return centers, gradients, velocities, bboxes, elem, det

        cx = 0.0
        cy = 0.0
        gx = 0.0
        gy = 0.0
        for i in range(node_count):
            node = connectivity[elem, i]
            cx += N[i] * local_coords[i, 0]
            cy += N[i] * local_coords[i, 1]
            gx += grad[0, i] * head[node]
            gy += grad[1, i] * head[node]
        pressure_head = avg_head if horizontal_pressure else avg_head - local_coords[0, 1]
        if model_codes[mat] == VGFLOW_MATERIAL_TABLE:
            _psi, _theta, _sat, kr, _capacity = vgflow_table_state_numba(
                table_values,
                table_offsets[mat],
                table_offsets[mat + 1],
                pressure_head,
                theta_r[mat],
                theta_s[mat],
            )
        else:
            _theta, _sat, kr, _capacity = vgflow_water_state_numba(
                model_codes[mat],
                pressure_head,
                alpha[mat],
                n_values[mat],
                theta_r[mat],
                theta_s[mat],
            )

        c = math.cos(angle_rad[mat])
        s = math.sin(angle_rad[mat])
        kxx = (kx[mat] * c * c + ky[mat] * s * s) * kr
        kxy = ((kx[mat] - ky[mat]) * c * s) * kr
        kyy = (kx[mat] * s * s + ky[mat] * c * c) * kr
        vx = -(kxx * gx + kxy * gy)
        vy = -(kxy * gx + kyy * gy)

        centers[elem, 0] = cx
        centers[elem, 1] = cy
        gradients[elem, 0] = gx
        gradients[elem, 1] = gy
        velocities[elem, 0] = vx
        velocities[elem, 1] = vy
        bboxes[elem, 0] = min_x
        bboxes[elem, 1] = min_y
        bboxes[elem, 2] = max_x
        bboxes[elem, 3] = max_y
    return centers, gradients, velocities, bboxes, -1, 0.0


@njit(cache=True)
def vgflow_contour_segments_numba(
    coords: np.ndarray,
    corner_connectivity: np.ndarray,
    corner_counts: np.ndarray,
    values: np.ndarray,
    levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    element_count = corner_connectivity.shape[0]
    max_segments = max(1, element_count * levels.shape[0] * 2)
    level_indices = np.empty(max_segments, dtype=np.int64)
    element_indices = np.empty(max_segments, dtype=np.int64)
    p0 = np.empty((max_segments, 2), dtype=np.float64)
    p1 = np.empty((max_segments, 2), dtype=np.float64)
    hit_x = np.empty(4, dtype=np.float64)
    hit_y = np.empty(4, dtype=np.float64)
    out = 0

    for level_index in range(levels.shape[0]):
        level = levels[level_index]
        for elem in range(element_count):
            count = corner_counts[elem]
            hit_count = 0
            for edge in range(count):
                ia = corner_connectivity[elem, edge]
                ib = corner_connectivity[elem, 0] if edge == count - 1 else corner_connectivity[elem, edge + 1]
                va = values[ia]
                vb = values[ib]
                if (level - va) * (level - vb) > 0.0:
                    continue
                denom = vb - va
                if abs(denom) <= 1.0e-14:
                    continue
                t = (level - va) / denom
                if t < -1.0e-12 or t > 1.0 + 1.0e-12:
                    continue
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                x = coords[ia, 0] + t * (coords[ib, 0] - coords[ia, 0])
                y = coords[ia, 1] + t * (coords[ib, 1] - coords[ia, 1])
                duplicate = False
                for old in range(hit_count):
                    if abs(x - hit_x[old]) <= 1.0e-10 and abs(y - hit_y[old]) <= 1.0e-10:
                        duplicate = True
                        break
                if not duplicate and hit_count < 4:
                    hit_x[hit_count] = x
                    hit_y[hit_count] = y
                    hit_count += 1
            for hit in range(0, hit_count - 1, 2):
                level_indices[out] = level_index
                element_indices[out] = elem
                p0[out, 0] = hit_x[hit]
                p0[out, 1] = hit_y[hit]
                p1[out, 0] = hit_x[hit + 1]
                p1[out, 1] = hit_y[hit + 1]
                out += 1
    return level_indices, element_indices, p0, p1, out


__all__ = [
    "VGFLOW_MATERIAL_SATURATED",
    "VGFLOW_MATERIAL_TABLE",
    "VGFLOW_MATERIAL_VAN_GENUCHTEN",
    "vgflow_contour_segments_numba",
    "vgflow_quad4_assembly_triplets_numba",
    "vgflow_quad8_assembly_triplets_numba",
    "vgflow_post_element_fields_numba",
    "vgflow_table_state_array_numba",
    "vgflow_table_state_numba",
    "vgflow_tri3_assembly_triplets_numba",
    "vgflow_tri6_assembly_triplets_numba",
    "vgflow_water_state_array_numba",
    "vgflow_water_state_numba",
]
