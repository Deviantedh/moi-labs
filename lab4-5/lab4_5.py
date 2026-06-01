from __future__ import annotations

import argparse
import math
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import scenes


EPS = 1e-5
DEFAULT_SEED = 20260525


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float | "Vec3") -> "Vec3":
        if isinstance(value, Vec3):
            return Vec3(self.x * value.x, self.y * value.y, self.z * value.z)
        return Vec3(self.x * value, self.y * value, self.z * value)

    def __rmul__(self, value: float | "Vec3") -> "Vec3":
        return self.__mul__(value)

    def __truediv__(self, value: float) -> "Vec3":
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        value = self.length()
        if value < EPS:
            raise ValueError("Cannot normalize a zero vector")
        return self / value

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def clamp01(self) -> "Vec3":
        return Vec3(
            min(1.0, max(0.0, self.x)),
            min(1.0, max(0.0, self.y)),
            min(1.0, max(0.0, self.z)),
        )


BLACK = Vec3(0.0, 0.0, 0.0)
WHITE = Vec3(1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3


@dataclass(frozen=True)
class Material:
    name: str
    diffuse: Vec3 = BLACK
    mirror: Vec3 = BLACK
    emission: Vec3 = BLACK

    def validate(self) -> None:
        for channel_name, diffuse, mirror in (
            ("red", self.diffuse.x, self.mirror.x),
            ("green", self.diffuse.y, self.mirror.y),
            ("blue", self.diffuse.z, self.mirror.z),
        ):
            if diffuse < 0.0 or mirror < 0.0:
                raise ValueError(f"Negative reflection coefficient in {self.name}")
            if diffuse + mirror > 1.0 + 1e-9:
                raise ValueError(
                    f"Material {self.name} is not physical in {channel_name} channel"
                )

    def is_light(self) -> bool:
        return self.emission.max_component() > 0.0


@dataclass(frozen=True)
class Triangle:
    a: Vec3
    b: Vec3
    c: Vec3
    material: Material
    object_id: int = 0
    edge1: Vec3 = field(init=False, repr=False)
    edge2: Vec3 = field(init=False, repr=False)
    normal: Vec3 = field(init=False)
    area: float = field(init=False)

    def __post_init__(self) -> None:
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        cross = edge1.cross(edge2)
        object.__setattr__(self, "edge1", edge1)
        object.__setattr__(self, "edge2", edge2)
        object.__setattr__(self, "normal", cross.normalized())
        object.__setattr__(self, "area", 0.5 * cross.length())


@dataclass(frozen=True)
class Hit:
    t: float
    point: Vec3
    normal: Vec3
    triangle: Triangle


@dataclass(frozen=True)
class AABB:
    minimum: Vec3
    maximum: Vec3

    @staticmethod
    def from_triangle(triangle: Triangle) -> "AABB":
        padding = EPS
        return AABB(
            Vec3(
                min(triangle.a.x, triangle.b.x, triangle.c.x) - padding,
                min(triangle.a.y, triangle.b.y, triangle.c.y) - padding,
                min(triangle.a.z, triangle.b.z, triangle.c.z) - padding,
            ),
            Vec3(
                max(triangle.a.x, triangle.b.x, triangle.c.x) + padding,
                max(triangle.a.y, triangle.b.y, triangle.c.y) + padding,
                max(triangle.a.z, triangle.b.z, triangle.c.z) + padding,
            ),
        )

    @staticmethod
    def from_triangles(triangles: list[Triangle]) -> "AABB":
        bounds = AABB.from_triangle(triangles[0])
        for triangle in triangles[1:]:
            bounds = bounds.union(AABB.from_triangle(triangle))
        return bounds

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            Vec3(
                min(self.minimum.x, other.minimum.x),
                min(self.minimum.y, other.minimum.y),
                min(self.minimum.z, other.minimum.z),
            ),
            Vec3(
                max(self.maximum.x, other.maximum.x),
                max(self.maximum.y, other.maximum.y),
                max(self.maximum.z, other.maximum.z),
            ),
        )

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        for origin, direction, box_min, box_max in (
            (ray.origin.x, ray.direction.x, self.minimum.x, self.maximum.x),
            (ray.origin.y, ray.direction.y, self.minimum.y, self.maximum.y),
            (ray.origin.z, ray.direction.z, self.minimum.z, self.maximum.z),
        ):
            if abs(direction) < EPS:
                if origin < box_min or origin > box_max:
                    return False
                continue

            inv_d = 1.0 / direction
            t0 = (box_min - origin) * inv_d
            t1 = (box_max - origin) * inv_d
            if inv_d < 0.0:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_max < t_min:
                return False
        return True


@dataclass(frozen=True)
class BVHNode:
    bounds: AABB
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None
    triangles: tuple[Triangle, ...] = ()


class Camera:
    def __init__(
        self,
        position: Vec3,
        look_at: Vec3,
        up: Vec3,
        fov_degrees: float,
        aspect: float,
    ) -> None:
        self.position = position
        self.forward = (look_at - position).normalized()
        self.right = self.forward.cross(up).normalized()
        self.up = self.right.cross(self.forward).normalized()
        self.scale = math.tan(math.radians(fov_degrees) * 0.5)
        self.aspect = aspect

    def generate_ray(self, x: float, y: float) -> Ray:
        px = (2.0 * x - 1.0) * self.aspect * self.scale
        py = (1.0 - 2.0 * y) * self.scale
        direction = (self.forward + self.right * px + self.up * py).normalized()
        return Ray(self.position, direction)


class Scene:
    def __init__(self, triangles: list[Triangle]) -> None:
        self.triangles = triangles
        self.bvh = build_bvh(triangles)
        self.lights = [triangle for triangle in triangles if triangle.material.is_light()]
        if not self.lights:
            raise ValueError("Scene must contain at least one emissive triangle")

        self.light_weights = []
        total_weight = 0.0
        for light in self.lights:
            weight = luminance(light.material.emission) * light.area
            self.light_weights.append(weight)
            total_weight += weight
        self.total_light_weight = total_weight


def luminance(color: Vec3) -> float:
    return 0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z


def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    return (direction - normal * (2.0 * direction.dot(normal))).normalized()


def offset_ray_origin(point: Vec3, normal: Vec3, direction: Vec3) -> Vec3:
    sign = 1.0 if normal.dot(direction) >= 0.0 else -1.0
    return point + normal * (EPS * sign)


def triangle_centroid(triangle: Triangle) -> Vec3:
    return (triangle.a + triangle.b + triangle.c) / 3.0


def axis_value(point: Vec3, axis: int) -> float:
    if axis == 0:
        return point.x
    if axis == 1:
        return point.y
    return point.z


def build_bvh(triangles: list[Triangle], leaf_size: int = 4) -> BVHNode:
    bounds = AABB.from_triangles(triangles)
    if len(triangles) <= leaf_size:
        return BVHNode(bounds=bounds, triangles=tuple(triangles))

    extent = bounds.maximum - bounds.minimum
    if extent.x >= extent.y and extent.x >= extent.z:
        axis = 0
    elif extent.y >= extent.z:
        axis = 1
    else:
        axis = 2

    sorted_triangles = sorted(
        triangles,
        key=lambda triangle: axis_value(triangle_centroid(triangle), axis),
    )
    midpoint = len(sorted_triangles) // 2
    left = build_bvh(sorted_triangles[:midpoint], leaf_size)
    right = build_bvh(sorted_triangles[midpoint:], leaf_size)
    return BVHNode(bounds=bounds, left=left, right=right)


def intersect_triangle(ray: Ray, triangle: Triangle) -> Hit | None:
    edge1 = triangle.edge1
    edge2 = triangle.edge2
    pvec = ray.direction.cross(edge2)
    det = edge1.dot(pvec)

    if abs(det) < EPS:
        return None

    inv_det = 1.0 / det
    tvec = ray.origin - triangle.a
    u = tvec.dot(pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return None

    qvec = tvec.cross(edge1)
    v = ray.direction.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None

    t = edge2.dot(qvec) * inv_det
    if t <= EPS:
        return None

    point = ray.origin + ray.direction * t
    normal = triangle.normal
    if normal.dot(ray.direction) > 0.0:
        normal = normal * -1.0
    return Hit(t, point, normal, triangle)


def intersect_scene(ray: Ray, scene: Scene) -> Hit | None:
    return intersect_bvh(ray, scene.bvh, float("inf"))


def intersect_bvh(ray: Ray, node: BVHNode, closest_t: float) -> Hit | None:
    if not node.bounds.hit(ray, EPS, closest_t):
        return None

    if node.triangles:
        closest_hit = None
        for triangle in node.triangles:
            hit = intersect_triangle(ray, triangle)
            if hit is not None and hit.t < closest_t:
                closest_hit = hit
                closest_t = hit.t
        return closest_hit

    left_hit = intersect_bvh(ray, node.left, closest_t) if node.left is not None else None
    if left_hit is not None:
        closest_t = left_hit.t
    right_hit = intersect_bvh(ray, node.right, closest_t) if node.right is not None else None
    if right_hit is not None:
        return right_hit
    return left_hit


def occluded_bvh(ray: Ray, node: BVHNode, max_t: float, target_light: Triangle) -> bool:
    if not node.bounds.hit(ray, EPS, max_t):
        return False

    if node.triangles:
        for triangle in node.triangles:
            if triangle is target_light:
                continue
            hit = intersect_triangle(ray, triangle)
            if hit is not None and hit.t < max_t:
                return True
        return False

    if node.left is not None and occluded_bvh(ray, node.left, max_t, target_light):
        return True
    if node.right is not None and occluded_bvh(ray, node.right, max_t, target_light):
        return True
    return False


def make_basis(normal: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    n = normal.normalized()
    helper = Vec3(1.0, 0.0, 0.0) if abs(n.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
    tangent = helper.cross(n).normalized()
    bitangent = n.cross(tangent)
    return tangent, bitangent, n


def sample_cosine_hemisphere(normal: Vec3, rng: random.Random) -> Vec3:
    tangent, bitangent, n = make_basis(normal)
    r = math.sqrt(rng.random())
    phi = 2.0 * math.pi * rng.random()
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - r * r))
    return (tangent * x + bitangent * y + n * z).normalized()


def sample_triangle(triangle: Triangle, rng: random.Random) -> Vec3:
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    return triangle.a + (triangle.b - triangle.a) * u + (triangle.c - triangle.a) * v


def choose_light(scene: Scene, rng: random.Random) -> tuple[Triangle, float]:
    target = rng.random() * scene.total_light_weight
    accumulated = 0.0
    for light, weight in zip(scene.lights, scene.light_weights):
        accumulated += weight
        if target <= accumulated:
            selection_pdf = weight / scene.total_light_weight
            return light, selection_pdf
    light = scene.lights[-1]
    return light, scene.light_weights[-1] / scene.total_light_weight


def visible_to_light(
    point: Vec3,
    normal: Vec3,
    light_point: Vec3,
    light: Triangle,
    scene: Scene,
) -> bool:
    direction = light_point - point
    distance = direction.length()
    if distance <= EPS:
        return False

    direction = direction / distance
    shadow_origin = offset_ray_origin(point, normal, direction)
    shadow_ray = Ray(shadow_origin, direction)
    return not occluded_bvh(shadow_ray, scene.bvh, distance - 2.0 * EPS, light)


def estimate_direct_light_once(hit: Hit, scene: Scene, rng: random.Random) -> Vec3:
    material = hit.triangle.material
    if material.diffuse.max_component() <= 0.0:
        return BLACK

    light, selection_pdf = choose_light(scene, rng)
    light_point = sample_triangle(light, rng)
    to_light = light_point - hit.point
    distance_squared = to_light.dot(to_light)
    if distance_squared <= EPS:
        return BLACK

    light_dir = to_light.normalized()
    cos_surface = max(0.0, hit.normal.dot(light_dir))
    light_normal = light.normal
    cos_light = max(0.0, light_normal.dot(light_dir * -1.0))
    if cos_surface <= 0.0 or cos_light <= 0.0:
        return BLACK

    if not visible_to_light(hit.point, hit.normal, light_point, light, scene):
        return BLACK

    area_pdf = selection_pdf / light.area
    geometry = cos_surface * cos_light / distance_squared
    brdf = material.diffuse / math.pi
    return light.material.emission * brdf * (geometry / area_pdf)


def estimate_direct_light(hit: Hit, scene: Scene, rng: random.Random, light_samples: int) -> Vec3:
    if light_samples <= 1:
        return estimate_direct_light_once(hit, scene, rng)

    total = BLACK
    for _ in range(light_samples):
        total += estimate_direct_light_once(hit, scene, rng)
    return total / light_samples


def trace_ray(
    ray: Ray,
    scene: Scene,
    rng: random.Random,
    max_depth: int,
    light_samples: int,
) -> Vec3:
    radiance = BLACK
    throughput = WHITE

    for depth in range(max_depth):
        hit = intersect_scene(ray, scene)
        if hit is None:
            break

        material = hit.triangle.material
        if material.is_light():
            radiance += throughput * material.emission
            break

        radiance += throughput * estimate_direct_light(hit, scene, rng, light_samples)

        diffuse_probability = min(0.95, material.diffuse.max_component())
        mirror_probability = min(0.95 - diffuse_probability, material.mirror.max_component())
        event_probability = diffuse_probability + mirror_probability
        if event_probability <= 0.0:
            break

        event = rng.random()
        if event < diffuse_probability:
            new_direction = sample_cosine_hemisphere(hit.normal, rng)
            throughput = throughput * material.diffuse / diffuse_probability
        elif event < event_probability:
            new_direction = reflect(ray.direction, hit.normal)
            throughput = throughput * material.mirror / mirror_probability
        else:
            break

        if depth >= 3:
            survival = min(0.95, throughput.max_component())
            if rng.random() > survival:
                break
            throughput = throughput / survival

        ray = Ray(offset_ray_origin(hit.point, hit.normal, new_direction), new_direction)

    return radiance


def trace_ray_components(
    ray: Ray,
    scene: Scene,
    rng: random.Random,
    max_depth: int,
    light_samples: int,
) -> tuple[Vec3, Vec3]:
    direct = BLACK
    secondary = BLACK
    throughput = WHITE

    for depth in range(max_depth):
        hit = intersect_scene(ray, scene)
        if hit is None:
            break

        material = hit.triangle.material
        if material.is_light():
            if depth == 0:
                direct += throughput * material.emission
            else:
                secondary += throughput * material.emission
            break

        direct_light = throughput * estimate_direct_light(hit, scene, rng, light_samples)
        if depth == 0:
            direct += direct_light
        else:
            secondary += direct_light

        diffuse_probability = min(0.95, material.diffuse.max_component())
        mirror_probability = min(0.95 - diffuse_probability, material.mirror.max_component())
        event_probability = diffuse_probability + mirror_probability
        if event_probability <= 0.0:
            break

        event = rng.random()
        if event < diffuse_probability:
            new_direction = sample_cosine_hemisphere(hit.normal, rng)
            throughput = throughput * material.diffuse / diffuse_probability
        elif event < event_probability:
            new_direction = reflect(ray.direction, hit.normal)
            throughput = throughput * material.mirror / mirror_probability
        else:
            break

        if depth >= 3:
            survival = min(0.95, throughput.max_component())
            if rng.random() > survival:
                break
            throughput = throughput / survival

        ray = Ray(offset_ray_origin(hit.point, hit.normal, new_direction), new_direction)

    return direct, secondary


@dataclass
class RenderBuffers:
    direct: list[list[Vec3]]
    secondary: list[list[Vec3]]
    total: list[list[Vec3]]
    depth: list[list[float]]
    normal: list[list[Vec3]]
    object_id: list[list[int]]


def gamma_correct(color: Vec3, exposure: float, gamma: float) -> tuple[int, int, int]:
    mapped = (color * exposure).clamp01()
    inv_gamma = 1.0 / gamma
    return (
        int(255.0 * (mapped.x ** inv_gamma) + 0.5),
        int(255.0 * (mapped.y ** inv_gamma) + 0.5),
        int(255.0 * (mapped.z ** inv_gamma) + 0.5),
    )


def save_ppm(path: str, pixels: list[list[Vec3]], exposure: float, gamma: float) -> None:
    height = len(pixels)
    width = len(pixels[0])
    with open(path, "w", encoding="ascii") as file:
        file.write(f"P3\n{width} {height}\n255\n")
        for row in pixels:
            values = []
            for color in row:
                values.extend(str(value) for value in gamma_correct(color, exposure, gamma))
            file.write(" ".join(values) + "\n")


def save_png(path: str, pixels: list[list[Vec3]], exposure: float, gamma: float) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; PNG output is skipped.")
        return

    height = len(pixels)
    width = len(pixels[0])
    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            gamma_correct(pixels[y][x], exposure, gamma)
            for y in range(height)
            for x in range(width)
        ]
    )
    image.save(path)


def save_linear_png(path: str, pixels: list[list[Vec3]]) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; PNG output is skipped.")
        return

    height = len(pixels)
    width = len(pixels[0])
    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            (
                int(255.0 * min(1.0, max(0.0, pixels[y][x].x)) + 0.5),
                int(255.0 * min(1.0, max(0.0, pixels[y][x].y)) + 0.5),
                int(255.0 * min(1.0, max(0.0, pixels[y][x].z)) + 0.5),
            )
            for y in range(height)
            for x in range(width)
        ]
    )
    image.save(path)


def save_depth_png(path: str, depth: list[list[float]]) -> None:
    finite_values = [value for row in depth for value in row if math.isfinite(value)]
    if not finite_values:
        save_linear_png(path, [[BLACK for _ in row] for row in depth])
        return

    min_depth = min(finite_values)
    max_depth = max(finite_values)
    scale = max(max_depth - min_depth, EPS)
    pixels = []
    for row in depth:
        out_row = []
        for value in row:
            if math.isfinite(value):
                v = (value - min_depth) / scale
            else:
                v = 0.0
            out_row.append(Vec3(v, v, v))
        pixels.append(out_row)
    save_linear_png(path, pixels)


def save_normal_png(path: str, normal: list[list[Vec3]]) -> None:
    pixels = []
    for row in normal:
        pixels.append([Vec3(n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5) for n in row])
    save_linear_png(path, pixels)


def save_object_id_png(path: str, object_id: list[list[int]]) -> None:
    palette = [
        Vec3(0.12, 0.16, 0.22),
        Vec3(0.75, 0.16, 0.13),
        Vec3(0.16, 0.50, 0.72),
        Vec3(0.70, 0.70, 0.66),
        Vec3(0.85, 0.58, 0.25),
        Vec3(0.55, 0.55, 0.58),
        Vec3(0.80, 0.72, 0.28),
        Vec3(0.24, 0.72, 0.62),
        Vec3(0.72, 0.45, 0.72),
    ]
    pixels = []
    for row in object_id:
        pixels.append([palette[value % len(palette)] if value >= 0 else BLACK for value in row])
    save_linear_png(path, pixels)


def first_hit_buffers(ray: Ray, scene: Scene) -> tuple[float, Vec3, int]:
    hit = intersect_scene(ray, scene)
    if hit is None:
        return float("inf"), Vec3(0.0, 0.0, 1.0), -1
    return hit.t, hit.normal, hit.triangle.object_id


def render_buffers(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    samples_per_pixel: int,
    max_depth: int,
    light_samples: int,
    seed: int,
    progress_callback=None,
    chunk_callback=None,
    workers: int = 1,
    tile_size: int = 32,
) -> RenderBuffers:
    direct = [[BLACK for _ in range(width)] for _ in range(height)]
    secondary = [[BLACK for _ in range(width)] for _ in range(height)]
    total = [[BLACK for _ in range(width)] for _ in range(height)]
    depth = [[float("inf") for _ in range(width)] for _ in range(height)]
    normal = [[Vec3(0.0, 0.0, 1.0) for _ in range(width)] for _ in range(height)]
    object_id = [[-1 for _ in range(width)] for _ in range(height)]
    start = time.time()

    if workers > 1:
        completed_rows = 0
        chunk_size = max(1, min(tile_size, height))
        chunks = [(y, min(height, y + chunk_size)) for y in range(0, height, chunk_size)]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    render_buffer_row_range,
                    scene,
                    camera,
                    width,
                    height,
                    samples_per_pixel,
                    max_depth,
                    light_samples,
                    seed,
                    y_start,
                    y_end,
                )
                for y_start, y_end in chunks
            ]

            for future in as_completed(futures):
                y_start, direct_rows, secondary_rows, total_rows, depth_rows, normal_rows, object_id_rows = future.result()
                for offset in range(len(total_rows)):
                    row_index = y_start + offset
                    direct[row_index] = direct_rows[offset]
                    secondary[row_index] = secondary_rows[offset]
                    total[row_index] = total_rows[offset]
                    depth[row_index] = depth_rows[offset]
                    normal[row_index] = normal_rows[offset]
                    object_id[row_index] = object_id_rows[offset]

                completed_rows += len(total_rows)
                if chunk_callback is not None:
                    chunk_callback(y_start, total_rows)

                elapsed = time.time() - start
                print(f"Rendered buffers {completed_rows}/{height} rows in {elapsed:.1f} s")
                if progress_callback is not None:
                    progress_callback(completed_rows, height, elapsed)

        return RenderBuffers(direct, secondary, total, depth, normal, object_id)

    for y in range(height):
        for x in range(width):
            rng = random.Random(seed + y * width + x)
            direct_sum = BLACK
            secondary_sum = BLACK
            center_ray = camera.generate_ray((x + 0.5) / width, (y + 0.5) / height)
            depth[y][x], normal[y][x], object_id[y][x] = first_hit_buffers(center_ray, scene)

            for _ in range(samples_per_pixel):
                u = (x + rng.random()) / width
                v = (y + rng.random()) / height
                d, s = trace_ray_components(camera.generate_ray(u, v), scene, rng, max_depth, light_samples)
                direct_sum += d
                secondary_sum += s

            direct[y][x] = direct_sum / samples_per_pixel
            secondary[y][x] = secondary_sum / samples_per_pixel
            total[y][x] = direct[y][x] + secondary[y][x]

        if chunk_callback is not None:
            chunk_callback(y, [total[y]])

        if y == 0 or (y + 1) % max(1, height // 20) == 0 or y == height - 1:
            elapsed = time.time() - start
            print(f"Rendered buffers {y + 1}/{height} rows in {elapsed:.1f} s")
            if progress_callback is not None:
                progress_callback(y + 1, height, elapsed)

    return RenderBuffers(direct, secondary, total, depth, normal, object_id)


def render_buffer_row_range(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    samples_per_pixel: int,
    max_depth: int,
    light_samples: int,
    seed: int,
    y_start: int,
    y_end: int,
) -> tuple[
    int,
    list[list[Vec3]],
    list[list[Vec3]],
    list[list[Vec3]],
    list[list[float]],
    list[list[Vec3]],
    list[list[int]],
]:
    direct_rows = []
    secondary_rows = []
    total_rows = []
    depth_rows = []
    normal_rows = []
    object_id_rows = []

    for y in range(y_start, y_end):
        direct_row = []
        secondary_row = []
        total_row = []
        depth_row = []
        normal_row = []
        object_id_row = []

        for x in range(width):
            rng = random.Random(seed + y * width + x)
            direct_sum = BLACK
            secondary_sum = BLACK
            center_ray = camera.generate_ray((x + 0.5) / width, (y + 0.5) / height)
            hit_depth, hit_normal, hit_object_id = first_hit_buffers(center_ray, scene)

            for _ in range(samples_per_pixel):
                u = (x + rng.random()) / width
                v = (y + rng.random()) / height
                d, s = trace_ray_components(camera.generate_ray(u, v), scene, rng, max_depth, light_samples)
                direct_sum += d
                secondary_sum += s

            direct_value = direct_sum / samples_per_pixel
            secondary_value = secondary_sum / samples_per_pixel
            direct_row.append(direct_value)
            secondary_row.append(secondary_value)
            total_row.append(direct_value + secondary_value)
            depth_row.append(hit_depth)
            normal_row.append(hit_normal)
            object_id_row.append(hit_object_id)

        direct_rows.append(direct_row)
        secondary_rows.append(secondary_row)
        total_rows.append(total_row)
        depth_rows.append(depth_row)
        normal_rows.append(normal_row)
        object_id_rows.append(object_id_row)

    return y_start, direct_rows, secondary_rows, total_rows, depth_rows, normal_rows, object_id_rows


def gaussian_weight(value: float, sigma: float) -> float:
    return math.exp(-(value * value) / (2.0 * sigma * sigma))


def color_distance(a: Vec3, b: Vec3) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def filter_component(
    image: list[list[Vec3]],
    depth: list[list[float]],
    normal: list[list[Vec3]],
    object_id: list[list[int]],
    radius: int,
    sigma_spatial: float,
    sigma_color: float,
    sigma_depth: float,
    sigma_normal: float,
) -> list[list[Vec3]]:
    height = len(image)
    width = len(image[0])
    result = [[BLACK for _ in range(width)] for _ in range(height)]
    spatial = {}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial[(dy, dx)] = gaussian_weight(math.sqrt(dx * dx + dy * dy), sigma_spatial)

    for y in range(height):
        for x in range(width):
            current_object = object_id[y][x]
            if current_object < 0:
                result[y][x] = image[y][x]
                continue

            center_color = image[y][x]
            center_depth = depth[y][x]
            center_normal = normal[y][x]
            weighted_sum = BLACK
            weight_sum = 0.0

            for dy in range(-radius, radius + 1):
                yy = min(height - 1, max(0, y + dy))
                for dx in range(-radius, radius + 1):
                    xx = min(width - 1, max(0, x + dx))
                    if object_id[yy][xx] != current_object:
                        continue

                    weight = spatial[(dy, dx)]
                    weight *= gaussian_weight(color_distance(image[yy][xx], center_color), sigma_color)
                    if math.isfinite(center_depth) and math.isfinite(depth[yy][xx]):
                        weight *= gaussian_weight(abs(depth[yy][xx] - center_depth), sigma_depth)
                    weight *= gaussian_weight(color_distance(normal[yy][xx], center_normal), sigma_normal)

                    weighted_sum += image[yy][xx] * weight
                    weight_sum += weight

            result[y][x] = weighted_sum / weight_sum if weight_sum > EPS else center_color

    return result


def filter_component_row_range(
    image: list[list[Vec3]],
    depth: list[list[float]],
    normal: list[list[Vec3]],
    object_id: list[list[int]],
    radius: int,
    sigma_spatial: float,
    sigma_color: float,
    sigma_depth: float,
    sigma_normal: float,
    y_start: int,
    y_end: int,
) -> tuple[int, list[list[Vec3]]]:
    height = len(image)
    width = len(image[0])
    rows = []
    spatial = {}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial[(dy, dx)] = gaussian_weight(math.sqrt(dx * dx + dy * dy), sigma_spatial)

    for y in range(y_start, y_end):
        row = []
        for x in range(width):
            current_object = object_id[y][x]
            if current_object < 0:
                row.append(image[y][x])
                continue

            center_color = image[y][x]
            center_depth = depth[y][x]
            center_normal = normal[y][x]
            weighted_sum = BLACK
            weight_sum = 0.0

            for dy in range(-radius, radius + 1):
                yy = min(height - 1, max(0, y + dy))
                for dx in range(-radius, radius + 1):
                    xx = min(width - 1, max(0, x + dx))
                    if object_id[yy][xx] != current_object:
                        continue

                    weight = spatial[(dy, dx)]
                    weight *= gaussian_weight(color_distance(image[yy][xx], center_color), sigma_color)
                    if math.isfinite(center_depth) and math.isfinite(depth[yy][xx]):
                        weight *= gaussian_weight(abs(depth[yy][xx] - center_depth), sigma_depth)
                    weight *= gaussian_weight(color_distance(normal[yy][xx], center_normal), sigma_normal)

                    weighted_sum += image[yy][xx] * weight
                    weight_sum += weight

            row.append(weighted_sum / weight_sum if weight_sum > EPS else center_color)
        rows.append(row)
    return y_start, rows


def filter_component_parallel(
    image: list[list[Vec3]],
    depth: list[list[float]],
    normal: list[list[Vec3]],
    object_id: list[list[int]],
    radius: int,
    sigma_spatial: float,
    sigma_color: float,
    sigma_depth: float,
    sigma_normal: float,
    workers: int,
) -> list[list[Vec3]]:
    if workers <= 1:
        return filter_component(
            image,
            depth,
            normal,
            object_id,
            radius,
            sigma_spatial,
            sigma_color,
            sigma_depth,
            sigma_normal,
        )

    height = len(image)
    width = len(image[0])
    result = [[BLACK for _ in range(width)] for _ in range(height)]
    chunk_size = max(1, height // (workers * 4))
    chunks = [(y, min(height, y + chunk_size)) for y in range(0, height, chunk_size)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                filter_component_row_range,
                image,
                depth,
                normal,
                object_id,
                radius,
                sigma_spatial,
                sigma_color,
                sigma_depth,
                sigma_normal,
                y_start,
                y_end,
            )
            for y_start, y_end in chunks
        ]
        for future in as_completed(futures):
            y_start, rows = future.result()
            for offset, row in enumerate(rows):
                result[y_start + offset] = row
    return result


def preserve_luminance_by_object(
    filtered: list[list[Vec3]],
    reference: list[list[Vec3]],
    object_id: list[list[int]],
) -> list[list[Vec3]]:
    ref_sums: dict[int, float] = {}
    filtered_sums: dict[int, float] = {}
    height = len(filtered)
    width = len(filtered[0])

    for y in range(height):
        for x in range(width):
            oid = object_id[y][x]
            if oid < 0:
                continue
            ref_sums[oid] = ref_sums.get(oid, 0.0) + luminance(reference[y][x])
            filtered_sums[oid] = filtered_sums.get(oid, 0.0) + luminance(filtered[y][x])

    result = [[filtered[y][x] for x in range(width)] for y in range(height)]
    for y in range(height):
        for x in range(width):
            oid = object_id[y][x]
            if oid < 0:
                continue
            denom = filtered_sums.get(oid, 0.0)
            if denom > EPS:
                result[y][x] = filtered[y][x] * (ref_sums[oid] / denom)
    return result


def bilateral_filter_buffers(
    buffers: RenderBuffers,
    radius: int,
    iterations: int,
    sigma_spatial: float,
    sigma_color: float,
    sigma_depth: float,
    sigma_normal: float,
    workers: int,
) -> list[list[Vec3]]:
    direct = buffers.direct
    secondary = buffers.secondary
    for _ in range(iterations):
        direct = filter_component_parallel(
            direct,
            buffers.depth,
            buffers.normal,
            buffers.object_id,
            radius,
            sigma_spatial,
            sigma_color,
            sigma_depth,
            sigma_normal,
            workers,
        )
        secondary = filter_component_parallel(
            secondary,
            buffers.depth,
            buffers.normal,
            buffers.object_id,
            radius,
            sigma_spatial,
            sigma_color,
            sigma_depth,
            sigma_normal,
            workers,
        )

    combined = [
        [direct[y][x] + secondary[y][x] for x in range(len(direct[0]))]
        for y in range(len(direct))
    ]
    return preserve_luminance_by_object(combined, buffers.total, buffers.object_id)


def save_lab45_outputs(
    output_dir: str,
    buffers: RenderBuffers,
    filtered: list[list[Vec3]],
    exposure: float,
    gamma: float,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_png(os.path.join(output_dir, "render_noisy.png"), buffers.total, exposure, gamma)
    save_png(os.path.join(output_dir, "render_filtered.png"), filtered, exposure, gamma)
    save_png(os.path.join(output_dir, "direct.png"), buffers.direct, exposure, gamma)
    save_png(os.path.join(output_dir, "secondary.png"), buffers.secondary, exposure, gamma)
    save_depth_png(os.path.join(output_dir, "depth.png"), buffers.depth)
    save_normal_png(os.path.join(output_dir, "normal.png"), buffers.normal)
    save_object_id_png(os.path.join(output_dir, "object_id.png"), buffers.object_id)
    save_lab45_metrics(os.path.join(output_dir, "results_lr4_5.txt"), buffers, filtered)


def object_luminance_sums(image: list[list[Vec3]], object_id: list[list[int]]) -> dict[int, float]:
    sums: dict[int, float] = {}
    for y, row in enumerate(image):
        for x, color in enumerate(row):
            oid = object_id[y][x]
            if oid < 0:
                continue
            sums[oid] = sums.get(oid, 0.0) + luminance(color)
    return sums


def save_lab45_metrics(path: str, buffers: RenderBuffers, filtered: list[list[Vec3]]) -> None:
    noisy_sums = object_luminance_sums(buffers.total, buffers.object_id)
    filtered_sums = object_luminance_sums(filtered, buffers.object_id)
    lines = [
        "ЛР 5",
        "Билатеральная фильтрация результата path tracing",
        "",
        "Суммарная яркость по объектам:",
        "object_id; noisy; filtered; filtered-noisy",
    ]
    for oid in sorted(noisy_sums):
        lines.append(
            f"{oid}; {noisy_sums[oid]:.6f}; {filtered_sums.get(oid, 0.0):.6f}; "
            f"{filtered_sums.get(oid, 0.0) - noisy_sums[oid]:.6f}"
        )

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def render_row_range(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    samples_per_pixel: int,
    max_depth: int,
    light_samples: int,
    seed: int,
    y_start: int,
    y_end: int,
) -> tuple[int, list[list[Vec3]]]:
    rows = []
    for y in range(y_start, y_end):
        row = []
        for x in range(width):
            rng = random.Random(seed + y * width + x)
            color = BLACK
            for _ in range(samples_per_pixel):
                u = (x + rng.random()) / width
                v = (y + rng.random()) / height
                color += trace_ray(camera.generate_ray(u, v), scene, rng, max_depth, light_samples)
            row.append(color / samples_per_pixel)
        rows.append(row)
    return y_start, rows


def render(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    samples_per_pixel: int,
    max_depth: int,
    light_samples: int,
    seed: int,
    progress_callback=None,
    chunk_callback=None,
    workers: int = 1,
    tile_size: int = 32,
) -> list[list[Vec3]]:
    pixels = [[BLACK for _ in range(width)] for _ in range(height)]
    start = time.time()

    if workers > 1:
        completed_rows = 0
        chunk_size = max(1, min(tile_size, height))
        chunks = [(y, min(height, y + chunk_size)) for y in range(0, height, chunk_size)]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    render_row_range,
                    scene,
                    camera,
                    width,
                    height,
                    samples_per_pixel,
                    max_depth,
                    light_samples,
                    seed,
                    y_start,
                    y_end,
                )
                for y_start, y_end in chunks
            ]

            for future in as_completed(futures):
                y_start, rows = future.result()
                for offset, row in enumerate(rows):
                    pixels[y_start + offset] = row

                completed_rows += len(rows)
                elapsed = time.time() - start
                print(f"Rendered {completed_rows}/{height} rows in {elapsed:.1f} s")
                if chunk_callback is not None:
                    chunk_callback(y_start, rows)
                if progress_callback is not None:
                    progress_callback(completed_rows, height, elapsed)

        return pixels

    for y in range(height):
        _, rows = render_row_range(
            scene,
            camera,
            width,
            height,
            samples_per_pixel,
            max_depth,
            light_samples,
            seed,
            y,
            y + 1,
        )
        pixels[y] = rows[0]
        if chunk_callback is not None:
            chunk_callback(y, rows)

        if y == 0 or (y + 1) % max(1, height // 20) == 0 or y == height - 1:
            elapsed = time.time() - start
            print(f"Rendered {y + 1}/{height} rows in {elapsed:.1f} s")
            if progress_callback is not None:
                progress_callback(y + 1, height, elapsed)

    return pixels


def add_quad(
    triangles: list[Triangle],
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    material: Material,
    object_id: int,
) -> None:
    triangles.append(Triangle(a, b, c, material, object_id))
    triangles.append(Triangle(a, c, d, material, object_id))


def add_box(
    triangles: list[Triangle],
    minimum: Vec3,
    maximum: Vec3,
    material: Material,
    object_id: int,
) -> None:
    x0, y0, z0 = minimum.x, minimum.y, minimum.z
    x1, y1, z1 = maximum.x, maximum.y, maximum.z
    p000 = Vec3(x0, y0, z0)
    p001 = Vec3(x0, y0, z1)
    p010 = Vec3(x0, y1, z0)
    p011 = Vec3(x0, y1, z1)
    p100 = Vec3(x1, y0, z0)
    p101 = Vec3(x1, y0, z1)
    p110 = Vec3(x1, y1, z0)
    p111 = Vec3(x1, y1, z1)

    add_quad(triangles, p000, p001, p011, p010, material, object_id)
    add_quad(triangles, p100, p110, p111, p101, material, object_id)
    add_quad(triangles, p000, p100, p101, p001, material, object_id)
    add_quad(triangles, p010, p011, p111, p110, material, object_id)
    add_quad(triangles, p001, p101, p111, p011, material, object_id)
    add_quad(triangles, p000, p010, p110, p100, material, object_id)


def add_pyramid(
    triangles: list[Triangle],
    center: Vec3,
    size: float,
    height: float,
    material: Material,
    object_id: int,
) -> None:
    half = size * 0.5
    y = center.y
    p0 = Vec3(center.x - half, y, center.z - half)
    p1 = Vec3(center.x + half, y, center.z - half)
    p2 = Vec3(center.x + half, y, center.z + half)
    p3 = Vec3(center.x - half, y, center.z + half)
    top = Vec3(center.x, y + height, center.z)

    add_quad(triangles, p0, p1, p2, p3, material, object_id)
    triangles.append(Triangle(p0, top, p1, material, object_id))
    triangles.append(Triangle(p1, top, p2, material, object_id))
    triangles.append(Triangle(p2, top, p3, material, object_id))
    triangles.append(Triangle(p3, top, p0, material, object_id))


def build_cornell_box() -> Scene:
    white = Material("white diffuse", diffuse=Vec3(0.78, 0.78, 0.74))
    red = Material("red diffuse", diffuse=Vec3(0.75, 0.12, 0.10))
    blue_green = Material("blue green diffuse", diffuse=Vec3(0.10, 0.55, 0.62))
    mirror = Material("weak mirror", diffuse=Vec3(0.18, 0.18, 0.18), mirror=Vec3(0.55, 0.55, 0.55))
    light = Material("area light", emission=Vec3(12.0, 11.0, 9.0))

    for material in (white, red, blue_green, mirror, light):
        material.validate()

    triangles: list[Triangle] = []

    # Cornell box dimensions: x left/right, y height, z depth.
    add_quad(triangles, Vec3(-1, 0, -1), Vec3(1, 0, -1), Vec3(1, 0, 1), Vec3(-1, 0, 1), white, 1)
    add_quad(triangles, Vec3(-1, 2, -1), Vec3(-1, 2, 1), Vec3(1, 2, 1), Vec3(1, 2, -1), white, 2)
    add_quad(triangles, Vec3(-1, 0, 1), Vec3(1, 0, 1), Vec3(1, 2, 1), Vec3(-1, 2, 1), white, 3)
    add_quad(triangles, Vec3(-1, 0, -1), Vec3(-1, 0, 1), Vec3(-1, 2, 1), Vec3(-1, 2, -1), red, 4)
    add_quad(triangles, Vec3(1, 0, -1), Vec3(1, 2, -1), Vec3(1, 2, 1), Vec3(1, 0, 1), blue_green, 5)

    add_quad(
        triangles,
        Vec3(-0.33, 1.98, -0.25),
        Vec3(0.33, 1.98, -0.25),
        Vec3(0.33, 1.98, 0.25),
        Vec3(-0.33, 1.98, 0.25),
        light,
        6,
    )

    add_box(triangles, Vec3(-0.72, 0.0, -0.15), Vec3(-0.28, 0.85, 0.35), white, 7)
    add_box(triangles, Vec3(0.22, 0.0, -0.45), Vec3(0.68, 0.55, 0.05), mirror, 8)

    return Scene(triangles)


def build_light_gallery() -> Scene:
    wall = Material("matte concrete", diffuse=Vec3(0.60, 0.61, 0.58))
    floor = Material("dark stone", diffuse=Vec3(0.24, 0.25, 0.25))
    graphite = Material("graphite pedestal", diffuse=Vec3(0.12, 0.13, 0.14), mirror=Vec3(0.18, 0.18, 0.18))
    brass = Material("warm diffuse block", diffuse=Vec3(0.67, 0.45, 0.22))
    mirror = Material("mirror pyramid", diffuse=Vec3(0.04, 0.04, 0.045), mirror=Vec3(0.86, 0.86, 0.86))
    cyan_light = Material("cyan wall light", emission=Vec3(1.1, 4.4, 5.0))
    amber_light = Material("amber wall light", emission=Vec3(5.4, 2.8, 0.9))
    skylight = Material("soft ceiling light", emission=Vec3(4.2, 4.0, 3.6))

    for material in (wall, floor, graphite, brass, mirror, cyan_light, amber_light, skylight):
        material.validate()

    triangles: list[Triangle] = []

    # A neutral gallery room: open front, deep back wall, side light panels.
    add_quad(triangles, Vec3(-1.7, 0.0, -3.2), Vec3(1.7, 0.0, -3.2), Vec3(1.7, 0.0, 1.9), Vec3(-1.7, 0.0, 1.9), floor, 1)
    add_quad(triangles, Vec3(-1.7, 0.0, 1.9), Vec3(1.7, 0.0, 1.9), Vec3(1.7, 2.25, 1.9), Vec3(-1.7, 2.25, 1.9), wall, 2)
    add_quad(triangles, Vec3(-1.7, 0.0, -1.25), Vec3(-1.7, 0.0, 1.9), Vec3(-1.7, 2.25, 1.9), Vec3(-1.7, 2.25, -1.25), wall, 3)
    add_quad(triangles, Vec3(1.7, 0.0, -1.25), Vec3(1.7, 2.25, -1.25), Vec3(1.7, 2.25, 1.9), Vec3(1.7, 0.0, 1.9), wall, 4)
    add_quad(triangles, Vec3(-1.7, 2.25, -1.25), Vec3(-1.7, 2.25, 1.9), Vec3(1.7, 2.25, 1.9), Vec3(1.7, 2.25, -1.25), wall, 5)

    add_quad(
        triangles,
        Vec3(-1.68, 0.35, -0.75),
        Vec3(-1.68, 1.85, -0.75),
        Vec3(-1.68, 1.85, 0.85),
        Vec3(-1.68, 0.35, 0.85),
        cyan_light,
        6,
    )
    add_quad(
        triangles,
        Vec3(1.68, 0.35, -0.35),
        Vec3(1.68, 0.35, 1.25),
        Vec3(1.68, 1.85, 1.25),
        Vec3(1.68, 1.85, -0.35),
        amber_light,
        7,
    )
    add_quad(
        triangles,
        Vec3(-0.55, 2.23, 0.05),
        Vec3(0.55, 2.23, 0.05),
        Vec3(0.55, 2.23, 0.85),
        Vec3(-0.55, 2.23, 0.85),
        skylight,
        8,
    )

    add_box(triangles, Vec3(-1.05, 0.0, 0.2), Vec3(-0.55, 0.35, 0.75), graphite, 9)
    add_box(triangles, Vec3(0.72, 0.0, 0.55), Vec3(1.18, 0.72, 1.1), brass, 10)
    add_box(triangles, Vec3(-0.25, 0.0, -0.35), Vec3(0.25, 0.16, 0.15), graphite, 11)
    add_pyramid(triangles, Vec3(0.0, 0.16, -0.1), 0.95, 0.9, mirror, 12)

    return Scene(triangles)


def build_scene(scene_name: str, aspect: float) -> tuple[Scene, Camera]:
    return scenes.build_scene(scene_name, aspect, sys.modules[__name__])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laboratory work 5: bilateral filtering for path traced images."
    )
    parser.add_argument("--gui", action="store_true", help="Open graphical interface")
    parser.add_argument("--cli", action="store_true", help="Run render from command line")
    parser.add_argument("--width", type=int, default=500, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=500, help="Image height in pixels")
    parser.add_argument("--scene", choices=scenes.SCENE_NAMES, default="gallery", help="Scene preset")
    parser.add_argument("--samples", type=int, default=4, help="Rays per pixel")
    parser.add_argument("--depth", type=int, default=6, help="Maximum path length")
    parser.add_argument("--light-samples", type=int, default=2, help="Direct light samples per hit")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers; 0 means auto")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size for lab 4 compatible rendering")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--exposure", type=float, default=0.55, help="Linear exposure multiplier")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma correction value")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4_render.ppm"),
        help="Output PPM path",
    )
    parser.add_argument("--png", action="store_true", help="Also save PNG next to the PPM file")
    parser.add_argument("--lab5", action="store_true", help="Render G-buffers and apply lab 5 bilateral filter")
    parser.add_argument("--filter-radius", type=int, default=5, help="Lab 5 filter radius")
    parser.add_argument("--filter-iterations", type=int, default=3, help="Lab 5 filter passes")
    parser.add_argument("--sigma-spatial", type=float, default=4.0, help="Lab 5 spatial sigma")
    parser.add_argument("--sigma-color", type=float, default=2.0, help="Lab 5 color sigma")
    parser.add_argument("--sigma-depth", type=float, default=0.35, help="Lab 5 depth sigma")
    parser.add_argument("--sigma-normal", type=float, default=0.55, help="Lab 5 normal sigma")
    parser.add_argument(
        "--lab5-output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4_5_results"),
        help="Directory for lab 5 filtered outputs",
    )
    return parser.parse_args()


def render_from_args(
    args: argparse.Namespace,
    progress_callback=None,
    chunk_callback=None,
) -> tuple[str, str | None]:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Width and height must be positive")
    if args.samples <= 0:
        raise ValueError("Samples per pixel must be positive")
    if args.depth <= 0:
        raise ValueError("Maximum depth must be positive")
    if args.light_samples <= 0:
        raise ValueError("Light samples must be positive")
    if args.gamma <= 0.0:
        raise ValueError("Gamma must be positive")
    if args.workers < 0:
        raise ValueError("Workers must be zero or positive")
    if args.tile_size <= 0:
        raise ValueError("Tile size must be positive")
    if args.filter_radius < 1:
        raise ValueError("Filter radius must be positive")
    if args.filter_iterations < 1:
        raise ValueError("Filter iterations must be positive")

    scene, camera = build_scene(args.scene, args.width / args.height)
    if args.workers == 0:
        workers = max(1, min((os.cpu_count() or 2) - 1, 8))
    else:
        workers = args.workers

    print(
        "Starting render: "
        f"scene={args.scene}, {args.width}x{args.height}, samples={args.samples}, depth={args.depth}, "
        f"light_samples={args.light_samples}, triangles={len(scene.triangles)}, workers={workers}"
    )

    if args.lab5:
        print("Lab 5 mode: rendering direct/secondary/depth/normal/object buffers")
        buffers = render_buffers(
            scene,
            camera,
            args.width,
            args.height,
            args.samples,
            args.depth,
            args.light_samples,
            args.seed,
            progress_callback,
            chunk_callback,
            workers,
            args.tile_size,
        )
        if progress_callback is not None:
            progress_callback(args.height, args.height, time.time())
        if hasattr(args, "stage_callback") and args.stage_callback is not None:
            args.stage_callback("Фильтрация изображения...")
        filtered = bilateral_filter_buffers(
            buffers,
            args.filter_radius,
            args.filter_iterations,
            args.sigma_spatial,
            args.sigma_color,
            args.sigma_depth,
            args.sigma_normal,
            workers,
        )
        save_lab45_outputs(args.lab5_output_dir, buffers, filtered, args.exposure, args.gamma)
        print(f"Lab 4-5 outputs saved to {args.lab5_output_dir}")
        return os.path.join(args.lab5_output_dir, "render_noisy.png"), os.path.join(
            args.lab5_output_dir, "render_filtered.png"
        )

    pixels = render(
        scene,
        camera,
        args.width,
        args.height,
        args.samples,
        args.depth,
        args.light_samples,
        args.seed,
        progress_callback,
        chunk_callback,
        workers,
        args.tile_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_ppm(args.output, pixels, args.exposure, args.gamma)
    print(f"PPM image saved to {args.output}")

    png_path = None
    if args.png:
        png_path = os.path.splitext(args.output)[0] + ".png"
        save_png(png_path, pixels, args.exposure, args.gamma)
        print(f"PNG image saved to {png_path}")

    return args.output, png_path


class Lab5App:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk

        self.root = tk.Tk()
        self.root.title("Лабораторная работа 5 - фильтрация path tracing")
        self.root.geometry("1060x780")
        self.root.minsize(920, 680)

        self.events: queue.Queue[tuple] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.preview_image = None
        self.live_image = None
        self.live_args = None
        self.last_preview_update = 0.0

        self.width_var = tk.StringVar(value="500")
        self.height_var = tk.StringVar(value="500")
        self.scene_var = tk.StringVar(value="gallery")
        self.samples_var = tk.StringVar(value="8")
        self.depth_var = tk.StringVar(value="6")
        self.light_samples_var = tk.StringVar(value="2")
        self.workers_var = tk.StringVar(value=str(max(1, min((os.cpu_count() or 2) - 1, 8))))
        self.tile_size_var = tk.StringVar(value="32")
        self.exposure_var = tk.StringVar(value="0.55")
        self.gamma_var = tk.StringVar(value="2.2")
        self.seed_var = tk.StringVar(value=str(DEFAULT_SEED))
        self.filter_radius_var = tk.StringVar(value="5")
        self.filter_iterations_var = tk.StringVar(value="3")
        self.sigma_spatial_var = tk.StringVar(value="4.0")
        self.sigma_color_var = tk.StringVar(value="2.0")
        self.sigma_depth_var = tk.StringVar(value="0.35")
        self.sigma_normal_var = tk.StringVar(value="0.55")
        self.output_dir_var = tk.StringVar(
            value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab5_results")
        )
        self.status_var = tk.StringVar(value="Готово к рендеру и фильтрации")

        self.build_widgets()
        self.root.after(100, self.process_events)

    def build_widgets(self) -> None:
        tk = self.tk
        ttk = self.ttk

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(main, text="Параметры ЛР 5", padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        self.add_entry(controls, "Ширина", self.width_var, 0)
        self.add_entry(controls, "Высота", self.height_var, 1)
        ttk.Label(controls, text="Сцена").grid(row=2, column=0, sticky=tk.W, pady=3)
        scene_box = ttk.Combobox(
            controls,
            textvariable=self.scene_var,
            values=scenes.SCENE_NAMES,
            state="readonly",
            width=12,
        )
        scene_box.grid(row=2, column=1, sticky=tk.EW, pady=3)
        self.add_entry(controls, "Лучи на пиксель", self.samples_var, 3)
        self.add_entry(controls, "Глубина луча", self.depth_var, 4)
        self.add_entry(controls, "Сэмплы света", self.light_samples_var, 5)
        self.add_entry(controls, "Потоки", self.workers_var, 6)
        self.add_entry(controls, "Размер тайла", self.tile_size_var, 7)
        self.add_entry(controls, "Экспозиция", self.exposure_var, 8)
        self.add_entry(controls, "Гамма", self.gamma_var, 9)
        self.add_entry(controls, "Seed", self.seed_var, 10)
        self.add_entry(controls, "Радиус фильтра", self.filter_radius_var, 11)
        self.add_entry(controls, "Проходы фильтра", self.filter_iterations_var, 12)
        self.add_entry(controls, "Sigma spatial", self.sigma_spatial_var, 13)
        self.add_entry(controls, "Sigma color", self.sigma_color_var, 14)
        self.add_entry(controls, "Sigma depth", self.sigma_depth_var, 15)
        self.add_entry(controls, "Sigma normal", self.sigma_normal_var, 16)

        ttk.Label(controls, text="Папка результатов").grid(row=17, column=0, columnspan=2, sticky=tk.W)
        output_row = ttk.Frame(controls)
        output_row.grid(row=18, column=0, columnspan=2, sticky=tk.EW, pady=(2, 8))
        output_row.columnconfigure(0, weight=1)
        ttk.Entry(output_row, textvariable=self.output_dir_var, width=34).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(output_row, text="...", width=4, command=self.choose_output).grid(row=0, column=1, padx=(6, 0))

        self.progress = ttk.Progressbar(controls, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.progress.grid(row=19, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))

        self.render_button = ttk.Button(controls, text="Запустить ЛР 5", command=self.start_render)
        self.render_button.grid(row=20, column=0, columnspan=2, sticky=tk.EW)

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=260,
            justify=tk.LEFT,
        ).grid(row=21, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))

        for column in range(2):
            controls.columnconfigure(column, weight=1)

        preview_frame = ttk.LabelFrame(main, text="Предпросмотр", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))
        self.preview_label = ttk.Label(
            preview_frame,
            text="После запуска здесь появится отфильтрованный результат",
            anchor=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def add_entry(self, parent, label: str, variable, row: int) -> None:
        ttk = self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=self.tk.W, pady=3)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky=self.tk.EW, pady=3)

    def choose_output(self) -> None:
        path = self.filedialog.askdirectory(
            title="Выбрать папку результатов",
            initialdir=os.path.dirname(os.path.abspath(self.output_dir_var.get())),
        )
        if path:
            self.output_dir_var.set(path)

    def read_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            gui=False,
            cli=True,
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            scene=self.scene_var.get(),
            samples=int(self.samples_var.get()),
            depth=int(self.depth_var.get()),
            light_samples=int(self.light_samples_var.get()),
            workers=int(self.workers_var.get()),
            tile_size=int(self.tile_size_var.get()),
            seed=int(self.seed_var.get()),
            exposure=float(self.exposure_var.get()),
            gamma=float(self.gamma_var.get()),
            output=os.path.join(self.output_dir_var.get(), "unused.ppm"),
            png=True,
            lab5=True,
            filter_radius=int(self.filter_radius_var.get()),
            filter_iterations=int(self.filter_iterations_var.get()),
            sigma_spatial=float(self.sigma_spatial_var.get()),
            sigma_color=float(self.sigma_color_var.get()),
            sigma_depth=float(self.sigma_depth_var.get()),
            sigma_normal=float(self.sigma_normal_var.get()),
            lab5_output_dir=self.output_dir_var.get(),
        )

    def start_render(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.messagebox.showinfo("Рендер уже идет", "Дождитесь завершения текущего рендера.")
            return

        try:
            args = self.read_args()
        except ValueError as exc:
            self.messagebox.showerror("Ошибка параметров", f"Проверьте числовые поля: {exc}")
            return

        self.progress["value"] = 0
        self.render_button["state"] = self.tk.DISABLED
        self.status_var.set("Рендер и фильтрация запущены...")
        self.prepare_live_preview(args)

        self.worker = threading.Thread(target=self.render_worker, args=(args,), daemon=True)
        self.worker.start()

    def render_worker(self, args: argparse.Namespace) -> None:
        try:
            def progress(row: int, total: int, elapsed: float) -> None:
                self.events.put(("progress", row, total, elapsed))

            def chunk(y_start: int, rows: list[list[Vec3]]) -> None:
                self.events.put(("chunk", y_start, rows))

            args.stage_callback = lambda text: self.events.put(("stage", text))
            ppm_path, png_path = render_from_args(args, progress, chunk)
            self.events.put(("done", ppm_path, png_path))
        except Exception as exc:
            self.events.put(("error", exc))

    def process_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, row, total, elapsed = event
                    percent = 100.0 * row / total
                    self.progress["value"] = percent
                    self.status_var.set(f"Готово {row}/{total} строк ({percent:.1f}%), {elapsed:.1f} c")
                elif kind == "chunk":
                    _, y_start, rows = event
                    self.apply_live_chunk(y_start, rows)
                elif kind == "stage":
                    _, text = event
                    self.status_var.set(text)
                elif kind == "done":
                    _, ppm_path, png_path = event
                    self.progress["value"] = 100
                    self.render_button["state"] = self.tk.NORMAL
                    self.refresh_live_preview(force=True)
                    if png_path:
                        self.show_preview(png_path)
                    self.status_var.set(f"Готово. Результаты: {os.path.dirname(png_path or ppm_path)}")
                elif kind == "error":
                    _, exc = event
                    self.render_button["state"] = self.tk.NORMAL
                    self.status_var.set("Рендер остановлен из-за ошибки")
                    self.messagebox.showerror("Ошибка рендера", str(exc))
        except queue.Empty:
            pass

        self.root.after(100, self.process_events)

    def prepare_live_preview(self, args: argparse.Namespace) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.live_image = None
            self.live_args = None
            self.preview_label.configure(text="Pillow не установлен, live preview недоступен")
            return

        self.live_args = args
        self.live_image = Image.new("RGB", (args.width, args.height), (0, 0, 0))
        self.last_preview_update = 0.0
        self.refresh_live_preview(force=True)

    def apply_live_chunk(self, y_start: int, rows: list[list[Vec3]]) -> None:
        if self.live_image is None or self.live_args is None:
            return

        from PIL import Image

        for offset, row in enumerate(rows):
            pixels = [
                gamma_correct(color, self.live_args.exposure, self.live_args.gamma)
                for color in row
            ]
            row_image = Image.new("RGB", (self.live_args.width, 1))
            row_image.putdata(pixels)
            self.live_image.paste(row_image, (0, y_start + offset))

        self.refresh_live_preview(force=False)

    def refresh_live_preview(self, force: bool) -> None:
        if self.live_image is None:
            return

        now = time.time()
        if not force and now - self.last_preview_update < 0.35:
            return

        try:
            from PIL import Image, ImageTk
        except ImportError:
            return

        image = self.live_image.copy()
        image.thumbnail((720, 720), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")
        self.last_preview_update = now

    def show_preview(self, path: str) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self.preview_label.configure(text="Pillow не установлен, предпросмотр PNG недоступен")
            return

        image = Image.open(path).copy()
        image.thumbnail((720, 720), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    if args.gui or (not args.cli and len(sys.argv) == 1):
        Lab5App().run()
    else:
        render_from_args(args)


if __name__ == "__main__":
    main()
