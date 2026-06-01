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


def render_tile(
    scene: Scene,
    camera: Camera,
    width: int,
    height: int,
    samples_per_pixel: int,
    max_depth: int,
    light_samples: int,
    seed: int,
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
) -> tuple[int, int, list[list[Vec3]]]:
    rows = []
    for y in range(y_start, y_end):
        row = []
        for x in range(x_start, x_end):
            rng = random.Random(seed + y * width + x)
            color = BLACK
            for _ in range(samples_per_pixel):
                u = (x + rng.random()) / width
                v = (y + rng.random()) / height
                color += trace_ray(camera.generate_ray(u, v), scene, rng, max_depth, light_samples)
            row.append(color / samples_per_pixel)
        rows.append(row)
    return x_start, y_start, rows


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
    tile_size = max(1, tile_size)
    tiles = [
        (x, y, min(width, x + tile_size), min(height, y + tile_size))
        for y in range(0, height, tile_size)
        for x in range(0, width, tile_size)
    ]
    total_tiles = len(tiles)

    if workers > 1:
        completed_tiles = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    render_tile,
                    scene,
                    camera,
                    width,
                    height,
                    samples_per_pixel,
                    max_depth,
                    light_samples,
                    seed,
                    x_start,
                    y_start,
                    x_end,
                    y_end,
                )
                for x_start, y_start, x_end, y_end in tiles
            ]

            for future in as_completed(futures):
                x_start, y_start, rows = future.result()
                for offset, row in enumerate(rows):
                    y = y_start + offset
                    pixels[y][x_start : x_start + len(row)] = row

                completed_tiles += 1
                elapsed = time.time() - start
                print(f"Rendered {completed_tiles}/{total_tiles} tiles in {elapsed:.1f} s")
                if chunk_callback is not None:
                    chunk_callback(x_start, y_start, rows)
                if progress_callback is not None:
                    progress_callback(completed_tiles, total_tiles, elapsed)

        return pixels

    for completed_tiles, (x_start, y_start, x_end, y_end) in enumerate(tiles, start=1):
        _, _, rows = render_tile(
            scene,
            camera,
            width,
            height,
            samples_per_pixel,
            max_depth,
            light_samples,
            seed,
            x_start,
            y_start,
            x_end,
            y_end,
        )
        for offset, row in enumerate(rows):
            y = y_start + offset
            pixels[y][x_start : x_start + len(row)] = row
        if chunk_callback is not None:
            chunk_callback(x_start, y_start, rows)

        if completed_tiles == 1 or completed_tiles % max(1, total_tiles // 20) == 0 or completed_tiles == total_tiles:
            elapsed = time.time() - start
            print(f"Rendered {completed_tiles}/{total_tiles} tiles in {elapsed:.1f} s")
            if progress_callback is not None:
                progress_callback(completed_tiles, total_tiles, elapsed)

    return pixels


def build_scene(scene_name: str, aspect: float) -> tuple[Scene, Camera]:
    return scenes.build_scene(scene_name, aspect, sys.modules[__name__])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Laboratory work 4 draft: simple path tracer with global illumination."
    )
    parser.add_argument("--gui", action="store_true", help="Open graphical interface")
    parser.add_argument("--cli", action="store_true", help="Run render from command line")
    parser.add_argument("--width", type=int, default=500, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=500, help="Image height in pixels")
    parser.add_argument(
        "--scene",
        choices=scenes.SCENE_NAMES,
        default="gallery",
        help="Scene preset",
    )
    parser.add_argument("--samples", type=int, default=4, help="Rays per pixel")
    parser.add_argument("--depth", type=int, default=6, help="Maximum path length")
    parser.add_argument("--light-samples", type=int, default=2, help="Direct light samples per hit")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers; 0 means auto")
    parser.add_argument("--tile-size", type=int, default=32, help="Tile size for parallel rendering")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--exposure", type=float, default=0.55, help="Linear exposure multiplier")
    parser.add_argument("--gamma", type=float, default=2.2, help="Gamma correction value")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4_render.ppm"),
        help="Output PPM path",
    )
    parser.add_argument("--png", action="store_true", help="Also save PNG next to the PPM file")
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

    scene, camera = build_scene(args.scene, args.width / args.height)
    if args.workers == 0:
        workers = max(1, min((os.cpu_count() or 2) - 1, 8))
    else:
        workers = args.workers

    print(
        "Starting render: "
        f"scene={args.scene}, {args.width}x{args.height}, samples={args.samples}, "
        f"depth={args.depth}, light_samples={args.light_samples}, "
        f"triangles={len(scene.triangles)}, workers={workers}"
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


class Lab4App:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.ttk = ttk

        self.root = tk.Tk()
        self.root.title("Лабораторная работа 4 - Path tracing")
        self.root.geometry("980x720")
        self.root.minsize(860, 620)

        self.events: queue.Queue[tuple] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.preview_image = None
        self.live_image = None
        self.live_args = None
        self.last_preview_update = 0.0

        self.width_var = tk.StringVar(value="500")
        self.height_var = tk.StringVar(value="500")
        self.scene_var = tk.StringVar(value="gallery")
        self.samples_var = tk.StringVar(value="4")
        self.depth_var = tk.StringVar(value="6")
        self.light_samples_var = tk.StringVar(value="2")
        self.workers_var = tk.StringVar(value=str(max(1, min((os.cpu_count() or 2) - 1, 8))))
        self.tile_size_var = tk.StringVar(value="32")
        self.exposure_var = tk.StringVar(value="0.55")
        self.gamma_var = tk.StringVar(value="2.2")
        self.seed_var = tk.StringVar(value=str(DEFAULT_SEED))
        self.output_var = tk.StringVar(
            value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab4_render.ppm")
        )
        self.save_png_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Готово к рендеру")

        self.build_widgets()
        self.root.after(100, self.process_events)

    def build_widgets(self) -> None:
        tk = self.tk
        ttk = self.ttk

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(main, text="Параметры рендера", padding=10)
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

        ttk.Checkbutton(
            controls,
            text="Сохранять PNG для предпросмотра",
            variable=self.save_png_var,
        ).grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(8, 4))

        ttk.Label(controls, text="Файл PPM").grid(row=12, column=0, columnspan=2, sticky=tk.W)
        output_row = ttk.Frame(controls)
        output_row.grid(row=13, column=0, columnspan=2, sticky=tk.EW, pady=(2, 8))
        output_row.columnconfigure(0, weight=1)
        ttk.Entry(output_row, textvariable=self.output_var, width=34).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(output_row, text="...", width=4, command=self.choose_output).grid(row=0, column=1, padx=(6, 0))

        self.progress = ttk.Progressbar(controls, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.progress.grid(row=14, column=0, columnspan=2, sticky=tk.EW, pady=(4, 8))

        self.render_button = ttk.Button(controls, text="Запустить рендер", command=self.start_render)
        self.render_button.grid(row=15, column=0, columnspan=2, sticky=tk.EW)

        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=260,
            justify=tk.LEFT,
        ).grid(row=16, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))

        for column in range(2):
            controls.columnconfigure(column, weight=1)

        preview_frame = ttk.LabelFrame(main, text="Предпросмотр", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))
        self.preview_label = ttk.Label(
            preview_frame,
            text="После рендера здесь появится PNG-предпросмотр",
            anchor=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def add_entry(self, parent, label: str, variable, row: int) -> None:
        ttk = self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=self.tk.W, pady=3)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky=self.tk.EW, pady=3)

    def choose_output(self) -> None:
        path = self.filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".ppm",
            filetypes=(("PPM image", "*.ppm"), ("All files", "*.*")),
            initialfile=os.path.basename(self.output_var.get()),
            initialdir=os.path.dirname(os.path.abspath(self.output_var.get())),
        )
        if path:
            self.output_var.set(path)

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
            output=self.output_var.get(),
            png=bool(self.save_png_var.get()),
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
        self.status_var.set("Рендер запущен...")
        self.prepare_live_preview(args)

        self.worker = threading.Thread(target=self.render_worker, args=(args,), daemon=True)
        self.worker.start()

    def render_worker(self, args: argparse.Namespace) -> None:
        try:
            def progress(done: int, total: int, elapsed: float) -> None:
                self.events.put(("progress", done, total, elapsed))

            def chunk(x_start: int, y_start: int, rows: list[list[Vec3]]) -> None:
                self.events.put(("chunk", x_start, y_start, rows))

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
                    _, done, total, elapsed = event
                    percent = 100.0 * done / total
                    self.progress["value"] = percent
                    self.status_var.set(f"Готово {done}/{total} тайлов ({percent:.1f}%), {elapsed:.1f} c")
                elif kind == "chunk":
                    _, x_start, y_start, rows = event
                    self.apply_live_chunk(x_start, y_start, rows)
                elif kind == "done":
                    _, ppm_path, png_path = event
                    self.progress["value"] = 100
                    self.render_button["state"] = self.tk.NORMAL
                    self.refresh_live_preview(force=True)
                    if png_path:
                        self.show_preview(png_path)
                    self.status_var.set(f"Готово. PPM: {ppm_path}")
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

    def apply_live_chunk(self, x_start: int, y_start: int, rows: list[list[Vec3]]) -> None:
        if self.live_image is None or self.live_args is None:
            return

        from PIL import Image

        for offset, row in enumerate(rows):
            pixels = [
                gamma_correct(color, self.live_args.exposure, self.live_args.gamma)
                for color in row
            ]
            row_image = Image.new("RGB", (len(pixels), 1))
            row_image.putdata(pixels)
            self.live_image.paste(row_image, (x_start, y_start + offset))

        self.refresh_live_preview(force=False)

    def refresh_live_preview(self, force: bool) -> None:
        if self.live_image is None:
            return

        now = time.time()
        if not force and now - self.last_preview_update < 0.35:
            return

        try:
            from PIL import ImageTk
        except ImportError:
            return

        image = self.live_image.copy()
        image.thumbnail((650, 650))
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")
        self.last_preview_update = now

    def show_preview(self, path: str) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self.preview_label.configure(text="Pillow не установлен, предпросмотр PNG недоступен")
            return

        image = Image.open(path)
        image.thumbnail((650, 650), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_image, text="")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    if args.gui or (not args.cli and len(sys.argv) == 1):
        Lab4App().run()
    else:
        render_from_args(args)


if __name__ == "__main__":
    main()
