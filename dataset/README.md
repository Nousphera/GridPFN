# Datasets

Obtain each source directly from its provider and comply with its access and license terms. Preprocessing is implemented in [`gridpfn/core/utils/preprocess.py`](../gridpfn/core/utils/preprocess.py), and the training loader is
implemented in [`gridpfn/core/dataset.py`](../gridpfn/core/dataset.py).

| Source | Data used by this project | Selection |
|---|---|---|
| [Pecan Street Dataport](https://dataport.pecanstreet.org/) | Household appliance consumption and PV generation | New York, 25 homes, 1 May–31 October 2019, 15-minute data |
| [NYISO Day-Ahead Market LBMP - Zonal](https://www.nyiso.com/) | New York City zonal LBMP | `N.Y.C.` (PTID `61761`), 1 May–31 October 2019, hourly data |
| [NOAA NCEI Local Climatological Data](https://www.ncei.noaa.gov/) | New York Central Park dry-bulb temperature | Station `72505394728`, 1 May–31 October 2019, hourly observations |

## Obtain the source data

### Pecan Street Dataport

1. [Sign up for or log in to Dataport](https://dataport.pecanstreet.org/).
   Access requires approval or an appropriate license; university access may
   be available for qualifying academic use.
2. Accept and follow the applicable Dataport license terms.
3. Export the New York 15-minute electricity data for 1 May through 31 October
   2019 as one CSV file.
4. The final seasonal protocol uses all 25 homes: `27`, `142`, `387`, `558`, `914`, `950`, `1222`, `1240`, `1417`, `2096`, `2318`, `2358`, `3000`, `3488`, `3517`, `3700`, `3996`, `4283`, `4550`, `5058`, `5587`, `5679`, `5982`, `5997`, `9053`. See `configs/seasonal.json`; the production launcher uses the same cohort.

Processed files live at `dataset/split_homes_clean/home_ID.csv`; shared weather and prices live at `dataset/temp_price_newyork.csv`. The available period is May–October 2019. Model exports reference these local inputs rather than copying them into Git.

The preprocessing step groups the source appliances/rooms as follows:

| Output | Source |
|---|---|
| `ac (kWh)` | `air1`, `air2`, `airwindowunit1`, `housefan1` |
| `heater (kWh)` | `heater1`, `heater2`, `heater3`, `furnace1`, `furnace2` |
| `ev (kWh)` | `car1`, `car2` |
| `wm (kWh)` | `clotheswasher1`, `clotheswasher_dryg1`, `drye1`, `dishwasher1` |
| `pv (kWh)` | `solar`, `solar2` |
| `fixed_load (kWh)` | `bathroom1`, `bedroom1`, `diningroom1`, `livingroom1`, `office1`, `utilityroom1`, `garage1`, `kitchen1`, `kitchenapp1`, `kitchenapp2`, `disposal1`, `microwave1`, `oven1`, `range1`, `venthood1`, `freezer1`, `refrigerator1`, `lights_plugs1`, `lights_plugs2`, `lights_plugs3`, `lights_plugs4`, `pump1`, `sewerpump1`, `sumppump1`, `wellpump1`, `waterheater1`, `jacuzzi1`, `circpump1` |

### NYISO day-ahead prices

1. Go to [NYISO Day-Ahead Market LBMP - Zonal](https://www.nyiso.com/).
2. Download the archived monthly CSV file for May through October 2019.
3. Combine the CSV rows, retain the `N.Y.C.` zone with PTID `61761`, and save
   one CSV containing at least `Time Stamp` and `LBMP ($/MWHr)`.
4. Preprocessing parses the timestamps and converts prices from `$/MWh` to `$/kWh`.

### NOAA NCEI outdoor temperature

1. Go to [NOAA NCEI Local Climatological Data](https://www.ncei.noaa.gov/).
2. Select 1 May through 31 October 2019 and search for the New York Central
   Park station (`72505394728`).
3. Download CSV output containing `DATE` and `HourlyDryBulbTemperature`.
4. Preprocessing averages duplicate timestamps and converts degrees Fahrenheit to degrees Celsius.

## Public demonstration and release status

The generated-data assistant demonstration (`python -m energy_assistant prepare`) uses artificial households and
weather/price traces from explicit formulas and a fixed seed. No private data,
fitted household statistics or downloaded observations enter that generator.
Generated data and recorded real-household results are labeled separately.

The historical input CSVs remain local and untracked pending confirmation of
redistribution rights. The main real-data figures require authorized originals;
the synthetic demo does not reproduce those numerical findings. Older Git
history still contains the CSVs, so making this existing repository public
requires a rights decision or a separately sanitized export.
