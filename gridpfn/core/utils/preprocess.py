
import argparse
from pathlib import Path

import pandas as pd

from gridpfn.paths import ROOT

APPLIANCE_GROUPS = {
    "ac (kWh)": ["air1", "air2", "airwindowunit1", "housefan1"],
    "heater (kWh)": ["heater1", "heater2", "heater3", "furnace1", "furnace2"],
    "ev (kWh)": ["car1", "car2"],
    "wm (kWh)": ["clotheswasher1", "clotheswasher_dryg1", "drye1", "dishwasher1"],
    "pv (kWh)": ["solar", "solar2"],
    "fixed_load (kWh)": [
        "bathroom1",
        "bedroom1",
        "diningroom1",
        "livingroom1",
        "office1",
        "utilityroom1",
        "garage1",
        "kitchen1",
        "kitchenapp1",
        "kitchenapp2",
        "disposal1",
        "microwave1",
        "oven1",
        "range1",
        "venthood1",
        "freezer1",
        "refrigerator1",
        "lights_plugs1",
        "lights_plugs2",
        "lights_plugs3",
        "lights_plugs4",
        "pump1",
        "sewerpump1",
        "sumppump1",
        "wellpump1",
        "waterheater1",
        "jacuzzi1",
        "circpump1",
    ],
}
DEVICE_COLUMNS = {device for devices in APPLIANCE_GROUPS.values() for device in devices}
SOURCE_COLUMNS = {"datetime", *DEVICE_COLUMNS}


def split_pecan_street_homes(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)
    data["_sort_time"] = pd.to_datetime(data["local_15min"], utc=True)
    output_paths = []

    for dataid, home_data in data.groupby("dataid", sort=True):
        home_data = home_data.sort_values("_sort_time", kind="stable").copy()
        home_data["local_15min"] = home_data["local_15min"].str.replace(
            r"(?:[+-]\d{2}(?::?\d{2})?|Z)$", "", regex=True
        )
        home_data = home_data.drop(columns=["dataid", "_sort_time", "grid", "leg1v", "leg2v"])
        home_data = home_data.rename(columns={"local_15min": "datetime"})
        home_data = home_data.dropna(axis="columns", how="all")
        output_path = output_dir / f"home_{dataid}.csv"
        home_data.to_csv(output_path, index=False, float_format="%.3f")
        output_paths.append(output_path)
    return output_paths


def create_clean_home_data(data_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    home_paths = sorted(
        data_dir.glob("home_*.csv"), key=lambda path: int(path.stem.removeprefix("home_"))
    )
    output_paths = []

    for home_path in home_paths:
        home_data = pd.read_csv(home_path, usecols=lambda column: column in SOURCE_COLUMNS)
        clean_data = pd.DataFrame({"datetime": home_data["datetime"]})
        for appliance, devices in APPLIANCE_GROUPS.items():
            available_devices = home_data.columns.intersection(devices)
            clean_data[appliance] = home_data[available_devices].sum(axis=1)
        output_path = output_dir / home_path.name
        clean_data.to_csv(output_path, index=False, float_format="%.3f")
        output_paths.append(output_path)
    return output_paths


def create_temp_price_newyork(
    weather_path,
    price_path,
    output_path,
    figure_path,
    start_time="2019-05-01 00:00:00",
    end_time="2019-10-31 23:45:00",
    frequency="15min",
):
    import matplotlib.pyplot as plt

    from gridpfn.core.utils.plots import configure_plot_style

    configure_plot_style()

    output_path = Path(output_path)
    figure_path = Path(figure_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(start_time, end_time, freq=frequency)
    weather = pd.read_csv(weather_path, usecols=["DATE", "HourlyDryBulbTemperature"])
    weather["DATE"] = pd.to_datetime(weather["DATE"])
    temperature = (weather.groupby("DATE")["HourlyDryBulbTemperature"].mean() - 32) * 5 / 9
    prices = pd.read_csv(price_path, usecols=["Time Stamp", "LBMP ($/MWHr)"])
    prices["Time Stamp"] = pd.to_datetime(prices["Time Stamp"], format="%m/%d/%Y %H:%M")
    price = prices.set_index("Time Stamp")["LBMP ($/MWHr)"] / 1000
    temperature = (
        temperature.reindex(temperature.index.union(timestamps))
        .interpolate(method="time")
        .reindex(timestamps)
    )
    price = (
        price.reindex(price.index.union(timestamps)).interpolate(method="time").reindex(timestamps)
    )
    data = pd.DataFrame(
        {
            "datetime": timestamps,
            "price ($/kWh)": price.to_numpy(),
            "temp (C)": temperature.to_numpy().round(2),
        }
    )
    csv_data = data.copy()
    csv_data["price ($/kWh)"] = csv_data["price ($/kWh)"].map("{:.4f}".format)
    csv_data["temp (C)"] = csv_data["temp (C)"].map("{:.2f}".format)
    csv_data.to_csv(output_path, index=False, float_format="%.4f")

    figure, temperature_axis = plt.subplots(figsize=(14, 6))
    price_axis = temperature_axis.twinx()
    temperature_axis.plot(
        data["datetime"], data["temp (C)"], color="tab:red", linewidth=0.8, label="Temperature"
    )
    price_axis.plot(
        data["datetime"],
        data["price ($/kWh)"],
        color="tab:blue",
        linewidth=0.8,
        label="Day-ahead price",
    )
    temperature_axis.set_xlabel("Datetime")
    temperature_axis.set_ylabel(r"Temperature ($^\circ\mathrm{C}$)")
    price_axis.set_ylabel("Price ($/kWh)")
    temperature_axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return data


def report_home_appliance_statistics(data_dir, output_path, steps_per_day=96):
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    home_paths = sorted(
        data_dir.glob("home_*.csv"), key=lambda path: int(path.stem.removeprefix("home_"))
    )
    statistics_by_home = []

    for home_path in home_paths:
        home_data = pd.read_csv(home_path)
        timestamps = pd.to_datetime(home_data.pop("datetime"))
        dates = timestamps.dt.strftime("%Y-%m-%d").rename("date")
        day_coverage = dates.value_counts()
        complete_dates = day_coverage.index[day_coverage == steps_per_day]
        complete = dates.isin(complete_dates)
        daily = (
            home_data.loc[complete].groupby(dates.loc[complete]).agg(["min", "max", "mean", "std"])
        )
        daily.columns.names = ["appliance", "metric"]
        statistics = daily.stack(level="appliance", future_stack=True).reset_index()
        statistics.insert(0, "home", int(home_path.stem.removeprefix("home_")))
        statistics_by_home.append(statistics)

    report = pd.concat(statistics_by_home, ignore_index=True)
    report.to_csv(output_path, index=False, float_format="%.3f")
    return report


def plot_home_appliance_statistics(statistics, output_dir):
    import matplotlib.pyplot as plt

    from gridpfn.core.utils.plots import F_SIZE, configure_plot_style

    configure_plot_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_labels = {
        "min": "Minimum",
        "max": "Maximum",
        "mean": "Mean",
        "std": "Standard deviation",
    }
    daily_averages = statistics.groupby(["home", "appliance"], as_index=False)[
        list(metric_labels)
    ].mean()
    homes = sorted(daily_averages["home"].unique())
    appliances = list(APPLIANCE_GROUPS)
    output_paths = []

    for metric, label in metric_labels.items():
        output_path = output_dir / f"heatmap_{metric}.pdf"
        values = daily_averages.pivot(index="appliance", columns="home", values=metric).reindex(
            index=appliances, columns=homes
        )
        color_map = plt.get_cmap("viridis").copy()
        color_map.set_bad("#D9D9D9")
        figure, axis = plt.subplots(figsize=(18, 6), constrained_layout=True)
        image = axis.imshow(
            values.to_numpy(),
            aspect="auto",
            interpolation="none",
            cmap=color_map,
            vmin=values.min().min(),
            vmax=values.max().max(),
        )
        axis.set_title(f"{label} of appliance power per day", fontsize=F_SIZE + 1)
        axis.set_xlabel("Home ID", fontsize=F_SIZE - 1)
        axis.set_ylabel("Appliance", fontsize=F_SIZE - 1)
        axis.set_xticks(range(len(homes)), labels=homes)
        axis.tick_params(axis="x", labelrotation=90, labelsize=F_SIZE - 3)
        axis.set_yticks(range(len(appliances)), labels=appliances)
        axis.tick_params(axis="y", labelsize=F_SIZE - 3)
        color_bar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
        color_bar.set_label("Power (kW)", fontsize=F_SIZE + 1)
        color_bar.ax.tick_params(labelsize=F_SIZE - 1)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def main(args):
    if args.run_split:
        split_paths = split_pecan_street_homes(args.path_raw_data, args.path_split_homes)
        print(f"Created {len(split_paths)} raw home CSV files")
    if args.run_clean:
        clean_paths = create_clean_home_data(args.path_split_homes, args.path_clean_homes)
        print(f"Created {len(clean_paths)} clean home CSV files")
        statistics = report_home_appliance_statistics(
            args.path_clean_homes, args.path_statistics, args.steps_per_day
        )
        print(f"Created {len(statistics)} daily home-appliance statistics")
        plot_paths = plot_home_appliance_statistics(statistics, args.path_statistics_plots)
        print(f"Created {len(plot_paths)} daily statistics heatmaps")
    if args.run_temp_price:
        temp_price_data = create_temp_price_newyork(
            args.path_weather,
            args.path_price,
            args.path_temp_price,
            args.path_temp_price_figure,
            args.start_time,
            args.end_time,
            args.frequency,
        )
        print(f"Created {len(temp_price_data)} temperature-price rows")


if __name__ == "__main__":
    path_dataset = ROOT / "dataset" / "pecan_street"
    path_raw = path_dataset / "data_raw"
    path_clean = path_dataset / "data_clean"
    path_clean_homes = path_clean / "split_homes_clean"
    path_statistics = path_clean / "appliance_statistics"
    parser = argparse.ArgumentParser(
        description="Preprocess Pecan Street home, weather, and price datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split the combined home CSV.",
    )
    parser.add_argument(
        "--run_clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create clean homes and statistics.",
    )
    parser.add_argument(
        "--run_temp_price",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create temperature-price outputs.",
    )
    parser.add_argument(
        "--start_time", default="2019-05-01 00:00:00", help="Interpolation start timestamp."
    )
    parser.add_argument(
        "--end_time", default="2019-10-31 23:45:00", help="Interpolation end timestamp."
    )
    parser.add_argument("--frequency", default="15min", help="Interpolation frequency.")
    parser.add_argument("--steps_per_day", type=int, default=96, help="Samples in a complete day.")
    parser.add_argument(
        "--path_raw_data",
        default=str(path_raw / "15minute_data_newyork.csv"),
        help="Combined raw home CSV.",
    )
    parser.add_argument(
        "--path_weather",
        default=str(path_raw / "ncei_weather_data_newyork.csv"),
        help="Raw weather CSV.",
    )
    parser.add_argument(
        "--path_price",
        default=str(path_raw / "nyiso_day_ahead_price.csv"),
        help="Raw day-ahead price CSV.",
    )
    parser.add_argument(
        "--path_split_homes",
        default=str(path_raw / "split_homes_raw"),
        help="Split-home output directory.",
    )
    parser.add_argument(
        "--path_clean_homes", default=str(path_clean_homes), help="Clean-home output directory."
    )
    parser.add_argument(
        "--path_temp_price",
        default=str(path_clean / "temp_price_newyork.csv"),
        help="Temperature-price CSV path.",
    )
    parser.add_argument(
        "--path_temp_price_figure",
        default=str(path_clean / "temp_price_newyork.pdf"),
        help="Temperature-price PDF path.",
    )
    parser.add_argument(
        "--path_statistics",
        default=str(path_statistics / "appliance_statistics.csv"),
        help="Appliance statistics CSV path.",
    )
    parser.add_argument(
        "--path_statistics_plots", default=str(path_statistics), help="Statistics plot directory."
    )
    main(parser.parse_args())
