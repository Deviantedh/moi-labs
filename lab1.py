from math import pi
import tkinter as tk
from tkinter import filedialog, messagebox

EPS = 1e-9

# Вычисления

def add_vec(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub_vec(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale_vec(v, k):
    return (v[0] * k, v[1] * k, v[2] * k)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(v):
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def normalize(v):
    v_len = length(v)
    if v_len < EPS:
        raise ValueError("Нельзя нормализовать около-нулевой вектор")
    return scale_vec(v, 1.0 / v_len)


def add_color(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale_color(color, k):
    return (color[0] * k, color[1] * k, color[2] * k)


def multiply_colors(a, b):
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def clamp_non_negative(value):
    return max(0.0, value)


def get_surface_basis(surface):
    p0 = surface["p0"]
    p1 = surface["p1"]
    p2 = surface["p2"]
    e1 = normalize(sub_vec(p1, p0))
    e2 = normalize(sub_vec(p2, p0))
    # Нормаль задаем в стандартной ориентации по правилу правой руки.
    normal = normalize(cross(sub_vec(p1, p0), sub_vec(p2, p0)))
    return e1, e2, normal


def local_to_global(surface, x_local, y_local):
    p0 = surface["p0"]
    e1, e2, _ = get_surface_basis(surface)
    return add_vec(p0, add_vec(scale_vec(e1, x_local), scale_vec(e2, y_local)))


def vector_from_light_to_point(light, point):
    return sub_vec(point, light["position"])


def vector_from_point_to_light(point, light):
    return sub_vec(light["position"], point)


def get_emitted_intensity(light, light_to_point):
    light_direction = normalize(light_to_point)
    source_axis = normalize(light["axis"])
    cos_theta = clamp_non_negative(dot(light_direction, source_axis))
    return scale_color(light["intensity_axis"], cos_theta)


def get_illuminance(light, point, normal):
    source_to_point = vector_from_light_to_point(light, point)
    point_to_light = vector_from_point_to_light(point, light)
    r_squared = dot(source_to_point, source_to_point)
    if r_squared < EPS:
        raise ValueError("Точка находится там где свет")

    # Угол падения считаем между нормалью и направлением из точки к источнику.
    cos_alpha = clamp_non_negative(dot(normalize(point_to_light), normal))
    intensity = get_emitted_intensity(light, source_to_point)
    return scale_color(intensity, cos_alpha / r_squared)


def get_half_vector(view, point_to_light):
    return normalize(add_vec(normalize(view), normalize(point_to_light)))


def get_brdf(surface, normal, view, point_to_light):
    h = get_half_vector(view, point_to_light)
    kd = surface["kd"]
    ks = surface["ks"]
    ke = surface["ke"]
    color = surface["color"]
    specular = ks * (clamp_non_negative(dot(h, normal)) ** ke)
    return scale_color(color, kd + specular)


def get_luminance(surface, point, view, lights):
    _, _, normal = get_surface_basis(surface)
    total = (0.0, 0.0, 0.0)

    for light in lights:
        point_to_light = vector_from_point_to_light(point, light)
        illuminance = get_illuminance(light, point, normal)
        brdf = get_brdf(surface, normal, view, point_to_light)
        total = add_color(total, multiply_colors(illuminance, brdf))

    return scale_color(total, 1.0 / pi)


def calculate_grid(surface, lights, view, x_values, y_values):
    _, _, normal = get_surface_basis(surface)
    illuminance_tables = []
    for _ in lights:
        illuminance_tables.append({})

    global_points = {}
    luminance_table = {}

    for y in y_values:
        for x in x_values:
            point = local_to_global(surface, x, y)
            global_points[(x, y)] = point
            for light_index, light in enumerate(lights):
                illuminance_tables[light_index][(x, y)] = get_illuminance(light, point, normal)
            luminance_table[(x, y)] = get_luminance(surface, point, view, lights)

    return global_points, illuminance_tables, luminance_table


# Форматирование и подготовка текста

def fmt_vec(v):
    return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})"


def fmt_color(color):
    return f"({color[0]:.6f}, {color[1]:.6f}, {color[2]:.6f})"


def parse_triplet(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected three comma-separated numbers")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def parse_float_list(text):
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def create_example_data():
    lights = [
        {
            "intensity_axis": (180.0, 120.0, 90.0),
            "axis": (0.0, 0.0, -1.0),
            "position": (0.6, 0.8, 2.8),
        },
        {
            "intensity_axis": (90.0, 160.0, 210.0),
            "axis": (-0.2, -0.1, -1.0),
            "position": (1.8, 0.5, 2.3),
        },
    ]

    surface = {
        "p0": (0.0, 0.0, 0.0),
        "p1": (3.0, 0.0, 0.0),
        "p2": (0.0, 2.5, 0.0),
        "color": (0.72, 0.55, 0.38),
        "kd": 0.65,
        "ks": 0.25,
        "ke": 12.0,
    }

    view = (0.1, 0.2, -1.0)
    x_values = [0.2, 0.6, 1.0, 1.4, 1.8]
    y_values = [0.2, 0.5, 0.8, 1.1, 1.4]
    return lights, surface, view, x_values, y_values


def build_word_table(title, x_values, y_values, table_map, formatter):
    lines = [title, "y\\x;\t" + ";\t".join(f"{x:.2f}" for x in x_values)]
    for y in y_values:
        row = [f"{y:.2f}"]
        for x in x_values:
            row.append(formatter(table_map[(x, y)]))
        lines.append(";\t".join(row))
    return "\n".join(lines)


def build_global_points_table(x_values, y_values, global_points):
    return build_word_table("Глобальные координаты PT", x_values, y_values, global_points, fmt_vec)


def build_illuminance_table(index, x_values, y_values, illuminance_table):
    return build_word_table(
        f"E{index}(RGB, PT)",
        x_values,
        y_values,
        illuminance_table,
        fmt_color,
    )


def build_luminance_table(x_values, y_values, luminance_table):
    return build_word_table("L(RGB, PT, v)", x_values, y_values, luminance_table, fmt_color)


def save_output_file(path, blocks):
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n\n".join(blocks))


def build_input_summary(surface, lights, view, x_values, y_values):
    _, _, normal = get_surface_basis(surface)
    lines = ["Input data"]
    for i, light in enumerate(lights, start=1):
        lines.append(f"I0{i}(RGB) = {fmt_color(light['intensity_axis'])}")
    for i, light in enumerate(lights, start=1):
        lines.append(f"O{i} = {fmt_vec(normalize(light['axis']))}")
    for i, light in enumerate(lights, start=1):
        lines.append(f"PL{i} = {fmt_vec(light['position'])}")
    lines.append(f"P0 = {fmt_vec(surface['p0'])}")
    lines.append(f"P1 = {fmt_vec(surface['p1'])}")
    lines.append(f"P2 = {fmt_vec(surface['p2'])}")
    lines.append(f"N = {fmt_vec(normal)}")
    lines.append(f"V = {fmt_vec(normalize(view))}")
    lines.append(f"K(RGB) = {fmt_color(surface['color'])}")
    lines.append(f"kd = {surface['kd']:.4f}")
    lines.append(f"ks = {surface['ks']:.4f}")
    lines.append(f"ke = {surface['ke']:.4f}")
    lines.append("x = " + ", ".join(f"{value:.2f}" for value in x_values))
    lines.append("y = " + ", ".join(f"{value:.2f}" for value in y_values))
    return "\n".join(lines)


def build_output_blocks(surface, lights, view, x_values, y_values):
    global_points, illuminance_tables, luminance_table = calculate_grid(
        surface, lights, view, x_values, y_values
    )
    blocks = [build_global_points_table(x_values, y_values, global_points)]
    for index, illuminance_table in enumerate(illuminance_tables, start=1):
        blocks.append(build_illuminance_table(index, x_values, y_values, illuminance_table))
    blocks.append(build_luminance_table(x_values, y_values, luminance_table))
    return blocks


def build_full_output(surface, lights, view, x_values, y_values):
    summary = build_input_summary(surface, lights, view, x_values, y_values)
    blocks = build_output_blocks(surface, lights, view, x_values, y_values)
    return summary + "\n\n\n\n" + "\n\n".join(blocks), blocks


# Интерфейс: окно, ввод и сохранение

def launch_gui():
    default_lights, default_surface, default_view, default_x, default_y = create_example_data()

    root = tk.Tk()
    root.title("ЛР1 МОИ  расчет освещенности и яркости")
    root.geometry("1500x850")

    left = tk.Frame(root, padx=12, pady=12)
    left.pack(side=tk.LEFT, fill=tk.Y)
    right = tk.Frame(root, padx=12, pady=12)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    tk.Label(left, text="Входные данные", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 8))

    entries = {}

    def add_entry(parent, key, label, default_value):
        tk.Label(parent, text=label).pack(anchor="w")
        entry = tk.Entry(parent, width=42)
        entry.pack(anchor="w", pady=(0, 6))
        entry.insert(0, default_value)
        entries[key] = entry

    for index, light in enumerate(default_lights, start=1):
        section = tk.LabelFrame(left, text=f"Источник света {index}", padx=8, pady=8)
        section.pack(fill=tk.X, pady=(0, 8))
        add_entry(section, f"i0_{index}", f"I0{index}(RGB)", ", ".join(str(v) for v in light["intensity_axis"]))
        add_entry(section, f"o_{index}", f"O{index}", ", ".join(str(v) for v in light["axis"]))
        add_entry(section, f"pl_{index}", f"PL{index}", ", ".join(str(v) for v in light["position"]))

    surface_section = tk.LabelFrame(left, text="Треугольник и материал", padx=8, pady=8)
    surface_section.pack(fill=tk.X, pady=(0, 8))
    add_entry(surface_section, "p0", "P0", ", ".join(str(v) for v in default_surface["p0"]))
    add_entry(surface_section, "p1", "P1", ", ".join(str(v) for v in default_surface["p1"]))
    add_entry(surface_section, "p2", "P2", ", ".join(str(v) for v in default_surface["p2"]))
    add_entry(surface_section, "color", "K(RGB)", ", ".join(str(v) for v in default_surface["color"]))
    add_entry(surface_section, "kd", "kd", str(default_surface["kd"]))
    add_entry(surface_section, "ks", "ks", str(default_surface["ks"]))
    add_entry(surface_section, "ke", "ke", str(default_surface["ke"]))

    obs_section = tk.LabelFrame(left, text="Наблюдение и сетка", padx=8, pady=8)
    obs_section.pack(fill=tk.X, pady=(0, 8))
    add_entry(obs_section, "view", "V", ", ".join(str(v) for v in default_view))
    add_entry(obs_section, "x_values", "Список x", ", ".join(str(v) for v in default_x))
    add_entry(obs_section, "y_values", "Список y", ", ".join(str(v) for v in default_y))

    tk.Label(right, text="Результат", font=("Arial", 14, "bold")).pack(anchor="w")
    result_text = tk.Text(right, wrap="none", font=("Courier", 10))
    result_text.pack(fill=tk.BOTH, expand=True)
    y_scroll = tk.Scrollbar(right, orient="vertical", command=result_text.yview)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    result_text.configure(yscrollcommand=y_scroll.set)
    x_scroll = tk.Scrollbar(right, orient="horizontal", command=result_text.xview)
    x_scroll.pack(fill=tk.X)
    result_text.configure(xscrollcommand=x_scroll.set)

    def collect_data():
        lights = []
        for index in range(1, 3):
            lights.append(
                {
                    "intensity_axis": parse_triplet(entries[f"i0_{index}"].get()),
                    "axis": parse_triplet(entries[f"o_{index}"].get()),
                    "position": parse_triplet(entries[f"pl_{index}"].get()),
                }
            )

        surface = {
            "p0": parse_triplet(entries["p0"].get()),
            "p1": parse_triplet(entries["p1"].get()),
            "p2": parse_triplet(entries["p2"].get()),
            "color": parse_triplet(entries["color"].get()),
            "kd": float(entries["kd"].get()),
            "ks": float(entries["ks"].get()),
            "ke": float(entries["ke"].get()),
        }
        view = parse_triplet(entries["view"].get())
        x_values = parse_float_list(entries["x_values"].get())
        y_values = parse_float_list(entries["y_values"].get())
        return lights, surface, view, x_values, y_values

    current_blocks = {"value": []}

    def calculate_and_show():
        try:
            lights, surface, view, x_values, y_values = collect_data()
            full_output, blocks = build_full_output(surface, lights, view, x_values, y_values)
            current_blocks["value"] = blocks
            result_text.delete("1.0", tk.END)
            result_text.insert("1.0", full_output)
        except Exception as error:
            messagebox.showerror("Ошибка ввода", str(error))


    def save_result():
        if not current_blocks["value"]:
            calculate_and_show()
        path = filedialog.asksaveasfilename(
            title="Сохранить таблицы",
            defaultextension=".txt",
            initialfile="word_tables.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            save_output_file(path, current_blocks["value"])
            messagebox.showinfo("Сохранено", f"Таблицы сохранены в файл:\n{path}")

    buttons = tk.Frame(left, pady=8)
    buttons.pack(fill=tk.X)
    tk.Button(buttons, text="Посчитать", command=calculate_and_show, width=16).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(buttons, text="Сохранить таблицы", command=save_result, width=18).pack(side=tk.LEFT)

    calculate_and_show()
    root.mainloop()


# Запуск программы

def main():
    print("1  Использовать пример из файла")
    print("2  Открыть окно для ввода параметров")
    mode = input("Выберите режим [ ]: ").strip() or "1"

    if mode == "2":
        launch_gui()
        return
    else:
        lights, surface, view, x_values, y_values = create_example_data()

    full_output, output_blocks = build_full_output(surface, lights, view, x_values, y_values)
    print(full_output)

    output_path = "word_tables.txt"
    save_output_file(output_path, output_blocks)
    print(f"\nТаблицы также сохранены в файл: {output_path}")


if __name__ == "__main__":
    main()
