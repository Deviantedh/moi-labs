from __future__ import annotations

import math


SCENE_NAMES = ("gallery", "triangle_disco", "cornell")


class SceneBuilder:
    def __init__(self, core) -> None:
        self.Vec3 = core.Vec3
        self.Material = core.Material
        self.Triangle = core.Triangle
        self.Scene = core.Scene
        self.Camera = core.Camera

    def add_quad(self, triangles, a, b, c, d, material) -> None:
        triangles.append(self.Triangle(a, b, c, material))
        triangles.append(self.Triangle(a, c, d, material))

    def rotate_y(self, point, angle_degrees: float, center):
        Vec3 = self.Vec3
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = point.x - center.x
        z = point.z - center.z
        return Vec3(
            center.x + x * cos_a + z * sin_a,
            point.y,
            center.z - x * sin_a + z * cos_a,
        )

    def make_basis(self, normal):
        Vec3 = self.Vec3
        n = normal.normalized()
        helper = Vec3(1.0, 0.0, 0.0) if abs(n.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
        tangent = helper.cross(n).normalized()
        bitangent = n.cross(tangent)
        return tangent, bitangent, n

    def add_aimed_quad(self, triangles, center, target, width: float, height: float, material) -> None:
        normal = (target - center).normalized()
        tangent, bitangent, _ = self.make_basis(normal)
        half_w = width * 0.5
        half_h = height * 0.5
        a = center - tangent * half_w - bitangent * half_h
        b = center + tangent * half_w - bitangent * half_h
        c = center + tangent * half_w + bitangent * half_h
        d = center - tangent * half_w + bitangent * half_h
        self.add_quad(triangles, a, b, c, d, material)

    def add_box(self, triangles, minimum, maximum, material) -> None:
        Vec3 = self.Vec3
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

        self.add_quad(triangles, p000, p001, p011, p010, material)
        self.add_quad(triangles, p100, p110, p111, p101, material)
        self.add_quad(triangles, p000, p100, p101, p001, material)
        self.add_quad(triangles, p010, p011, p111, p110, material)
        self.add_quad(triangles, p001, p101, p111, p011, material)
        self.add_quad(triangles, p000, p010, p110, p100, material)

    def add_pyramid(self, triangles, center, size: float, height: float, material) -> None:
        Vec3 = self.Vec3
        half = size * 0.5
        y = center.y
        p0 = Vec3(center.x - half, y, center.z - half)
        p1 = Vec3(center.x + half, y, center.z - half)
        p2 = Vec3(center.x + half, y, center.z + half)
        p3 = Vec3(center.x - half, y, center.z + half)
        top = Vec3(center.x, y + height, center.z)

        self.add_quad(triangles, p0, p1, p2, p3, material)
        triangles.append(self.Triangle(p0, top, p1, material))
        triangles.append(self.Triangle(p1, top, p2, material))
        triangles.append(self.Triangle(p2, top, p3, material))
        triangles.append(self.Triangle(p3, top, p0, material))

    def add_triangle_plate(
        self,
        triangles,
        center,
        width: float,
        height: float,
        angle_degrees: float,
        lean: float,
        material,
    ) -> None:
        Vec3 = self.Vec3
        base_y = center.y
        p0 = Vec3(center.x - width * 0.5, base_y, center.z)
        p1 = Vec3(center.x + width * 0.5, base_y, center.z)
        p2 = Vec3(center.x + lean, base_y + height, center.z)
        p0 = self.rotate_y(p0, angle_degrees, center)
        p1 = self.rotate_y(p1, angle_degrees, center)
        p2 = self.rotate_y(p2, angle_degrees, center)
        triangles.append(self.Triangle(p0, p2, p1, material))

    def add_faceted_sphere(self, triangles, center, radius: float, rings: int, segments: int, material) -> None:
        Vec3 = self.Vec3
        points = []
        for ring in range(1, rings):
            theta = math.pi * ring / rings
            y = math.cos(theta) * radius
            r = math.sin(theta) * radius
            row = []
            for segment in range(segments):
                phi = 2.0 * math.pi * segment / segments
                row.append(Vec3(center.x + r * math.cos(phi), center.y + y, center.z + r * math.sin(phi)))
            points.append(row)

        top = Vec3(center.x, center.y + radius, center.z)
        bottom = Vec3(center.x, center.y - radius, center.z)

        for segment in range(segments):
            next_segment = (segment + 1) % segments
            triangles.append(self.Triangle(top, points[0][segment], points[0][next_segment], material))

        for ring in range(len(points) - 1):
            upper = points[ring]
            lower = points[ring + 1]
            for segment in range(segments):
                next_segment = (segment + 1) % segments
                triangles.append(self.Triangle(upper[segment], lower[segment], lower[next_segment], material))
                triangles.append(self.Triangle(upper[segment], lower[next_segment], upper[next_segment], material))

        last = points[-1]
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            triangles.append(self.Triangle(bottom, last[next_segment], last[segment], material))

    def build_cornell_box(self):
        Vec3 = self.Vec3
        Material = self.Material
        white = Material("white diffuse", diffuse=Vec3(0.78, 0.78, 0.74))
        red = Material("red diffuse", diffuse=Vec3(0.75, 0.12, 0.10))
        blue_green = Material("blue green diffuse", diffuse=Vec3(0.10, 0.55, 0.62))
        mirror = Material("weak mirror", diffuse=Vec3(0.18, 0.18, 0.18), mirror=Vec3(0.55, 0.55, 0.55))
        light = Material("area light", emission=Vec3(12.0, 11.0, 9.0))

        for material in (white, red, blue_green, mirror, light):
            material.validate()

        triangles = []
        self.add_quad(triangles, Vec3(-1, 0, -1), Vec3(1, 0, -1), Vec3(1, 0, 1), Vec3(-1, 0, 1), white)
        self.add_quad(triangles, Vec3(-1, 2, -1), Vec3(-1, 2, 1), Vec3(1, 2, 1), Vec3(1, 2, -1), white)
        self.add_quad(triangles, Vec3(-1, 0, 1), Vec3(1, 0, 1), Vec3(1, 2, 1), Vec3(-1, 2, 1), white)
        self.add_quad(triangles, Vec3(-1, 0, -1), Vec3(-1, 0, 1), Vec3(-1, 2, 1), Vec3(-1, 2, -1), red)
        self.add_quad(triangles, Vec3(1, 0, -1), Vec3(1, 2, -1), Vec3(1, 2, 1), Vec3(1, 0, 1), blue_green)
        self.add_quad(triangles, Vec3(-0.33, 1.98, -0.25), Vec3(0.33, 1.98, -0.25), Vec3(0.33, 1.98, 0.25), Vec3(-0.33, 1.98, 0.25), light)
        self.add_box(triangles, Vec3(-0.72, 0.0, -0.15), Vec3(-0.28, 0.85, 0.35), white)
        self.add_box(triangles, Vec3(0.22, 0.0, -0.45), Vec3(0.68, 0.55, 0.05), mirror)
        return self.Scene(triangles)

    def build_light_gallery(self):
        Vec3 = self.Vec3
        Material = self.Material
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

        triangles = []
        self.add_quad(triangles, Vec3(-1.7, 0.0, -3.2), Vec3(1.7, 0.0, -3.2), Vec3(1.7, 0.0, 1.9), Vec3(-1.7, 0.0, 1.9), floor)
        self.add_quad(triangles, Vec3(-1.7, 0.0, 1.9), Vec3(1.7, 0.0, 1.9), Vec3(1.7, 2.25, 1.9), Vec3(-1.7, 2.25, 1.9), wall)
        self.add_quad(triangles, Vec3(-1.7, 0.0, -1.25), Vec3(-1.7, 0.0, 1.9), Vec3(-1.7, 2.25, 1.9), Vec3(-1.7, 2.25, -1.25), wall)
        self.add_quad(triangles, Vec3(1.7, 0.0, -1.25), Vec3(1.7, 2.25, -1.25), Vec3(1.7, 2.25, 1.9), Vec3(1.7, 0.0, 1.9), wall)
        self.add_quad(triangles, Vec3(-1.7, 2.25, -1.25), Vec3(-1.7, 2.25, 1.9), Vec3(1.7, 2.25, 1.9), Vec3(1.7, 2.25, -1.25), wall)
        self.add_quad(triangles, Vec3(-1.68, 0.35, -0.75), Vec3(-1.68, 1.85, -0.75), Vec3(-1.68, 1.85, 0.85), Vec3(-1.68, 0.35, 0.85), cyan_light)
        self.add_quad(triangles, Vec3(1.68, 0.35, -0.35), Vec3(1.68, 0.35, 1.25), Vec3(1.68, 1.85, 1.25), Vec3(1.68, 1.85, -0.35), amber_light)
        self.add_quad(triangles, Vec3(-0.55, 2.23, 0.05), Vec3(0.55, 2.23, 0.05), Vec3(0.55, 2.23, 0.85), Vec3(-0.55, 2.23, 0.85), skylight)
        self.add_box(triangles, Vec3(-1.05, 0.0, 0.2), Vec3(-0.55, 0.35, 0.75), graphite)
        self.add_box(triangles, Vec3(0.72, 0.0, 0.55), Vec3(1.18, 0.72, 1.1), brass)
        self.add_box(triangles, Vec3(-0.25, 0.0, -0.35), Vec3(0.25, 0.16, 0.15), graphite)
        self.add_pyramid(triangles, Vec3(0.0, 0.16, -0.1), 0.95, 0.9, mirror)
        return self.Scene(triangles)

    def build_triangle_disco(self):
        Vec3 = self.Vec3
        Material = self.Material
        floor = Material("dark dance floor", diffuse=Vec3(0.08, 0.075, 0.10), mirror=Vec3(0.24, 0.24, 0.28))
        wall = Material("deep club wall", diffuse=Vec3(0.15, 0.12, 0.19))
        magenta = Material("magenta dancer", diffuse=Vec3(0.76, 0.16, 0.56), mirror=Vec3(0.10, 0.08, 0.10))
        lime = Material("lime dancer", diffuse=Vec3(0.35, 0.72, 0.22), mirror=Vec3(0.08, 0.10, 0.05))
        blue = Material("blue dancer", diffuse=Vec3(0.12, 0.32, 0.78), mirror=Vec3(0.07, 0.09, 0.14))
        orange = Material("orange dancer", diffuse=Vec3(0.82, 0.40, 0.10), mirror=Vec3(0.08, 0.06, 0.04))
        mirror = Material("faceted mirror disco ball", diffuse=Vec3(0.08, 0.08, 0.09), mirror=Vec3(0.91, 0.91, 0.90))
        magenta_spot = Material("magenta aimed lamp", emission=Vec3(22.0, 2.4, 15.0))
        cyan_spot = Material("cyan aimed lamp", emission=Vec3(2.2, 15.0, 22.0))
        violet_spot = Material("violet corner lamp", emission=Vec3(9.0, 2.5, 18.0))
        teal_spot = Material("teal corner lamp", emission=Vec3(2.2, 16.0, 10.0))

        for material in (floor, wall, magenta, lime, blue, orange, mirror, magenta_spot, cyan_spot, violet_spot, teal_spot):
            material.validate()

        triangles = []
        self.add_quad(triangles, Vec3(-2.2, 0.0, -2.8), Vec3(2.2, 0.0, -2.8), Vec3(2.2, 0.0, 2.0), Vec3(-2.2, 0.0, 2.0), floor)
        self.add_quad(triangles, Vec3(-2.2, 0.0, 2.0), Vec3(2.2, 0.0, 2.0), Vec3(2.2, 2.45, 2.0), Vec3(-2.2, 2.45, 2.0), wall)
        self.add_quad(triangles, Vec3(-2.2, 0.0, -2.8), Vec3(-2.2, 0.0, 2.0), Vec3(-2.2, 2.45, 2.0), Vec3(-2.2, 2.45, -2.8), wall)
        self.add_quad(triangles, Vec3(2.2, 0.0, -2.8), Vec3(2.2, 2.45, -2.8), Vec3(2.2, 2.45, 2.0), Vec3(2.2, 0.0, 2.0), wall)
        self.add_quad(triangles, Vec3(-2.2, 2.45, -2.8), Vec3(-2.2, 2.45, 2.0), Vec3(2.2, 2.45, 2.0), Vec3(2.2, 2.45, -2.8), wall)

        ball_center = Vec3(0.0, 1.74, 0.20)
        self.add_aimed_quad(triangles, Vec3(-1.70, 2.05, -1.25), ball_center, 0.72, 0.52, magenta_spot)
        self.add_aimed_quad(triangles, Vec3(1.70, 1.95, -1.15), ball_center, 0.72, 0.52, cyan_spot)
        self.add_aimed_quad(triangles, Vec3(-1.78, 1.92, 1.55), ball_center, 0.56, 0.46, violet_spot)
        self.add_aimed_quad(triangles, Vec3(1.78, 1.88, 1.50), ball_center, 0.56, 0.46, teal_spot)

        self.add_triangle_plate(triangles, Vec3(-1.15, 0.0, -0.55), 0.72, 1.18, -16, -0.12, magenta)
        self.add_triangle_plate(triangles, Vec3(-0.35, 0.0, -0.22), 0.58, 0.98, 12, 0.12, lime)
        self.add_triangle_plate(triangles, Vec3(0.48, 0.0, -0.58), 0.70, 1.12, 24, -0.10, blue)
        self.add_triangle_plate(triangles, Vec3(1.16, 0.0, -0.04), 0.56, 0.92, -28, 0.10, orange)
        self.add_faceted_sphere(triangles, ball_center, 0.46, 20, 48, mirror)
        return self.Scene(triangles)

    def build_scene(self, scene_name: str, aspect: float):
        Vec3 = self.Vec3
        Camera = self.Camera
        if scene_name == "cornell":
            return (
                self.build_cornell_box(),
                Camera(Vec3(0.0, 1.0, -3.2), Vec3(0.0, 0.9, 0.15), Vec3(0.0, 1.0, 0.0), 42.0, aspect),
            )
        if scene_name == "gallery":
            return (
                self.build_light_gallery(),
                Camera(Vec3(0.0, 1.05, -3.75), Vec3(0.0, 0.8, 0.35), Vec3(0.0, 1.0, 0.0), 44.0, aspect),
            )
        if scene_name == "triangle_disco":
            return (
                self.build_triangle_disco(),
                Camera(Vec3(0.0, 1.02, -4.35), Vec3(0.0, 0.92, 0.15), Vec3(0.0, 1.0, 0.0), 46.0, aspect),
            )
        raise ValueError(f"Unknown scene: {scene_name}")


def build_scene(scene_name: str, aspect: float, core):
    return SceneBuilder(core).build_scene(scene_name, aspect)
