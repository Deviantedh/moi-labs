from math import sqrt
import os
import random

import matplotlib.pyplot as plt


# Лабораторная работа 2.
# Нужно найти интеграл функции f(x) = x^2 на отрезке [2, 5]
# разными вариантами метода Монте-Карло.

LEFT_BORDER = 2.0
RIGHT_BORDER = 5.0
SAMPLE_COUNTS = [100, 1000, 10000, 100000]
RANDOM_SEED = 2026323

# Точный ответ нужен, чтобы сравнивать с ним приближенные ответы.
TRUE_INTEGRAL = (RIGHT_BORDER**3 - LEFT_BORDER**3) / 3.0


def function_to_integrate(x):
    """Функция из задания: f(x) = x^2."""
    return x * x


def get_random_uniform_point():
    """Возвращает случайную точку с равномерным распределением на [2, 5]."""
    interval_length = RIGHT_BORDER - LEFT_BORDER
    return LEFT_BORDER + interval_length * random.random()


def calculate_simple_monte_carlo(sample_count):
    """
    Самый простой метод Монте-Карло.
    Берем случайные точки равномерно на [2, 5], считаем среднее f(x)
    и умножаем его на длину интервала.
    """
    values_sum = 0.0

    for _ in range(sample_count):
        x = get_random_uniform_point()
        values_sum += function_to_integrate(x)

    interval_length = RIGHT_BORDER - LEFT_BORDER
    average_value = values_sum / sample_count
    return interval_length * average_value


def calculate_stratified_monte_carlo(sample_count, step):
    """
    Метод со стратификацией.
    Интервал [2, 5] делится на маленькие интервалы-страты.
    В каждой страте отдельно запускается обычный Монте-Карло.
    """
    result = 0.0
    strata_count = round((RIGHT_BORDER - LEFT_BORDER) / step)
    samples_by_stratum = split_samples_between_parts(sample_count, strata_count)

    for stratum_number in range(strata_count):
        stratum_left = LEFT_BORDER + stratum_number * step
        stratum_right = min(stratum_left + step, RIGHT_BORDER)
        stratum_length = stratum_right - stratum_left
        samples_in_stratum = samples_by_stratum[stratum_number]

        values_sum = 0.0
        for _ in range(samples_in_stratum):
            x = stratum_left + stratum_length * random.random()
            values_sum += function_to_integrate(x)

        average_value = values_sum / samples_in_stratum
        result += stratum_length * average_value

    return result


def split_samples_between_parts(sample_count, parts_count):
    """Делит N выборок между частями так, чтобы сумма снова была ровно N."""
    base_count = sample_count // parts_count
    extra_samples = sample_count % parts_count

    samples_by_part = []
    for part_number in range(parts_count):
        if part_number < extra_samples:
            samples_by_part.append(base_count + 1)
        else:
            samples_by_part.append(base_count)

    return samples_by_part


def calculate_probability_density(x, power):
    """
    Плотность вероятности вида p(x) ~ x^power.
    Ее обязательно нужно нормировать, чтобы интеграл p(x) на [2, 5] был равен 1.
    """
    normalization = (power + 1) / (RIGHT_BORDER ** (power + 1) - LEFT_BORDER ** (power + 1))
    return normalization * x**power


def get_random_point_by_density(power):
    """
    Возвращает случайную точку с плотностью p(x) ~ x^power.
    Используется обратная функция распределения.
    """
    random_value = random.random()
    left_power = LEFT_BORDER ** (power + 1)
    right_power = RIGHT_BORDER ** (power + 1)
    return (left_power + random_value * (right_power - left_power)) ** (1 / (power + 1))


def calculate_importance_sampling(sample_count, power):
    """
    Выборка по значимости.
    Точки выбираются чаще там, где функция обычно дает больший вклад.
    Чтобы ответ не стал завышенным, каждый вклад делится на p(x).
    """
    values_sum = 0.0

    for _ in range(sample_count):
        x = get_random_point_by_density(power)
        density = calculate_probability_density(x, power)
        values_sum += function_to_integrate(x) / density

    return values_sum / sample_count


def calculate_mis_weight(x, selected_density_power, weight_variant):
    """
    Вес для многократной выборки по значимости.
    В задании используются две плотности: p1(x) ~ x и p2(x) ~ x^3.
    """
    p1 = calculate_probability_density(x, 1)
    p2 = calculate_probability_density(x, 3)

    if selected_density_power == 1:
        selected_density = p1
    else:
        selected_density = p2

    if weight_variant == 1:
        return selected_density / (p1 + p2)

    return selected_density**2 / (p1**2 + p2**2)


def calculate_multiple_importance_sampling(sample_count, weight_variant):
    """
    Многократная выборка по значимости.
    Половина точек генерируется по p1(x) ~ x,
    половина точек генерируется по p2(x) ~ x^3.
    Потом вклады смешиваются через веса.
    """
    first_density_samples = sample_count // 2
    second_density_samples = sample_count - first_density_samples

    first_sum = 0.0
    for _ in range(first_density_samples):
        x = get_random_point_by_density(1)
        weight = calculate_mis_weight(x, 1, weight_variant)
        density = calculate_probability_density(x, 1)
        first_sum += weight * function_to_integrate(x) / density

    second_sum = 0.0
    for _ in range(second_density_samples):
        x = get_random_point_by_density(3)
        weight = calculate_mis_weight(x, 3, weight_variant)
        density = calculate_probability_density(x, 3)
        second_sum += weight * function_to_integrate(x) / density

    return first_sum / first_density_samples + second_sum / second_density_samples


def calculate_russian_roulette(sample_count, survival_probability):
    """
    Русская рулетка.
    С вероятностью R вклад сохраняется и делится на R.
    С вероятностью 1 - R вклад отбрасывается.
    Так среднее значение не меняется, но разброс результата становится больше.
    """
    values_sum = 0.0

    for _ in range(sample_count):
        x = get_random_uniform_point()
        contribution = (RIGHT_BORDER - LEFT_BORDER) * function_to_integrate(x)

        if random.random() < survival_probability:
            values_sum += contribution / survival_probability

    return values_sum / sample_count


def make_result_row(method_name, parameter, sample_count, calculated_integral):
    """Готовит строку результата для Excel. Разделитель - точка с запятой."""
    absolute_error = abs(calculated_integral - TRUE_INTEGRAL)
    relative_error = absolute_error / TRUE_INTEGRAL
    estimated_error = TRUE_INTEGRAL / sqrt(sample_count)

    return (
        f"{method_name};"
        f"{parameter};"
        f"{sample_count};"
        f"{TRUE_INTEGRAL:.6f};"
        f"{calculated_integral:.6f};"
        f"{absolute_error:.6f};"
        f"{relative_error:.8f};"
        f"{estimated_error:.6f}"
    )


def add_result(rows, records, method_name, parameter, sample_count, calculated_integral):
    row = make_result_row(method_name, parameter, sample_count, calculated_integral)
    rows.append(row)
    records.append(
        {
            "method": method_name,
            "parameter": parameter,
            "sample_count": sample_count,
            "integral": calculated_integral,
            "absolute_error": abs(calculated_integral - TRUE_INTEGRAL),
        }
    )
    print(row)


def save_plots(records):
    """Сохраняет графики в PNG, чтобы потом вставить их в отчет вручную."""
    output_dir = "lab2_plots"
    os.makedirs(output_dir, exist_ok=True)

    methods = []
    for record in records:
        label = record["method"]
        if record["parameter"] != "-":
            label += ", " + record["parameter"]
        if label not in methods:
            methods.append(label)

    plt.figure(figsize=(12, 7))
    for label in methods:
        method_records = [
            record
            for record in records
            if label == record["method"] + ("" if record["parameter"] == "-" else ", " + record["parameter"])
        ]
        x_values = [record["sample_count"] for record in method_records]
        y_values = [record["integral"] for record in method_records]
        plt.plot(x_values, y_values, marker="o", label=label)

    plt.axhline(TRUE_INTEGRAL, color="black", linestyle="--", label="Истинный интеграл")
    plt.xscale("log")
    plt.xlabel("Размер выборки N")
    plt.ylabel("Оценка интеграла")
    plt.title("Сходимость оценок интеграла")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    for label in methods:
        method_records = [
            record
            for record in records
            if label == record["method"] + ("" if record["parameter"] == "-" else ", " + record["parameter"])
        ]
        x_values = [record["sample_count"] for record in method_records]
        # Нулевая ошибка не отображается на логарифмической оси, поэтому ставим малое число.
        y_values = [max(record["absolute_error"], 1e-12) for record in method_records]
        plt.plot(x_values, y_values, marker="o", label=label)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Размер выборки N")
    plt.ylabel("Абсолютная ошибка")
    plt.title("Абсолютная ошибка методов")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/absolute_error.png", dpi=200)
    plt.close()


def main():
    random.seed(RANDOM_SEED)

    rows = [
        "Метод;Параметр;N;Истинный интеграл;Вычисленный интеграл;Абсолютная погрешность;Относительная погрешность;Оценка Delta I"
    ]
    records = []
    print(rows[0])

    for sample_count in SAMPLE_COUNTS:
        add_result(
            rows,
            records,
            "Простой Монте-Карло",
            "-",
            sample_count,
            calculate_simple_monte_carlo(sample_count),
        )
        add_result(
            rows,
            records,
            "Стратификация",
            "шаг 1",
            sample_count,
            calculate_stratified_monte_carlo(sample_count, 1.0),
        )
        add_result(
            rows,
            records,
            "Стратификация",
            "шаг 0.5",
            sample_count,
            calculate_stratified_monte_carlo(sample_count, 0.5),
        )
        add_result(
            rows,
            records,
            "Выборка по значимости",
            "p(x) ~ x",
            sample_count,
            calculate_importance_sampling(sample_count, 1),
        )
        add_result(
            rows,
            records,
            "Выборка по значимости",
            "p(x) ~ x^2",
            sample_count,
            calculate_importance_sampling(sample_count, 2),
        )
        add_result(
            rows,
            records,
            "Выборка по значимости",
            "p(x) ~ x^3",
            sample_count,
            calculate_importance_sampling(sample_count, 3),
        )
        add_result(
            rows,
            records,
            "Многократная выборка по значимости",
            "веса p/(p1+p2)",
            sample_count,
            calculate_multiple_importance_sampling(sample_count, 1),
        )
        add_result(
            rows,
            records,
            "Многократная выборка по значимости",
            "веса p^2/(p1^2+p2^2)",
            sample_count,
            calculate_multiple_importance_sampling(sample_count, 2),
        )
        add_result(
            rows,
            records,
            "Русская рулетка",
            "R=0.5",
            sample_count,
            calculate_russian_roulette(sample_count, 0.5),
        )
        add_result(
            rows,
            records,
            "Русская рулетка",
            "R=0.75",
            sample_count,
            calculate_russian_roulette(sample_count, 0.75),
        )
        add_result(
            rows,
            records,
            "Русская рулетка",
            "R=0.95",
            sample_count,
            calculate_russian_roulette(sample_count, 0.95),
        )

    with open("results_lr2.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(rows) + "\n")

    save_plots(records)


if __name__ == "__main__":
    main()
