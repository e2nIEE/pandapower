# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Conversion of the JAO Static Grid Model excel/html data into a pandapower net."""

import json
import logging
import os

import pandas as pd

from pandapower.converter.jao._buses_lines import _create_buses_from_line_data, _create_lines
from pandapower.converter.jao._correction import _data_correction
from pandapower.converter.jao._geodata import _add_bus_geo, _parse_html_str
from pandapower.converter.jao._grid_groups import _invent_connections_between_grid_groups, drop_islanded_grid_groups
from pandapower.converter.jao._transformers import _create_transformers_and_buses
from pandapower.network import pandapowerNet
from pandapower.plotting import set_line_geodata_from_bus_geodata

logger = logging.getLogger(__name__)


def from_jao(
    excel_file_path: str,
    html_file_path: str | None,
    extend_data_for_grid_group_connections: bool,
    drop_grid_groups_islands: bool = False,
    apply_data_correction: bool = True,
    max_i_ka_fillna: float = 999,
    **kwargs,
) -> pandapowerNet:
    """
    Converts European (Core) EHV grid data provided by JAO (Joint Allocation Office), the
    "Single Allocation Platform (SAP) for all European Transmission System Operators (TSOs) that
    operate in accordance to EU legislation".

    **Data Sources and Availability:**
    The data are available at the website
    `JAO Static Grid Model <https://www.jao.eu/static-grid-model>`_ (November 2024).
    There, a map is provided to get an fine overview of the geographical extent and the scope of
    the data. These inlcude information about European (Core) lines, tielines, and transformers.

    **Limitations:**
    No information is available on load or generation.
    The data quality with regard to the interconnection of the equipment, the information provided
    and the (incomplete) geodata should be considered with caution.

    **Features of the converter:**
    - **Data Correction:** corrects known data inconsistencies, such as inconsistent spellings and missing necessary information.
    - **Geographical Data Parsing:** Parses geographical data from the HTML file to add geolocation information to buses and lines.
    - **Grid Group Connections:** Optionally extends the network by connecting islanded grid groups to avoid disconnected components.
    - **Data Customization:** Allows for customization through additional parameters to control transformer creation, grid group dropping, and voltage level deviations.

    Parameters:
        excel_file_path: input data including electrical parameters of grids' utilities, stored in multiple sheets
            of an excel file
        html_file_path: input data for geo information. If The converter should be run without geo information, None
            can be passed., provided by an html file
        extend_data_for_grid_group_connections: if True, connections (additional transformers and merging buses) are
            created to avoid islanded grid groups, by default False
        drop_grid_groups_islands: if True, islanded grid groups will be dropped if their number of buses is below
            `min_bus_number` default for this is 6 (default: False)
        apply_data_correction: _description_ (default: True)
        max_i_ka_fillna: value to fill missing values or data of false type in max_i_ka of lines and transformers.
            If no value should be set, you can also pass np.nan. (default: 999)

    Keyword Arguments:
        minimal_trafo_invention (Optional[bool]): applies if extend_data_for_grid_group_connections is True. Then,
            if minimal_trafo_invention is True, adding transformers stops when no grid groups is islanded anymore
            (does not apply for release version 5 or 6, i.e. it does not care what value is passed to
            minimal_trafo_invention). If False, all equally named buses that have different voltage level and lay in
            different groups will be connected via additional transformers (default: False)
        min_bus_number (Optional[int|str]): Threshold value to decide which small grid groups should be dropped and
            which large grid groups should be kept. If all islanded grid groups should be dropped except of the one
            largest, set "max". If all grid groups that do not contain a slack element should be dropped, set
            "unsupplied". (default: 6)
        rel_deviation_threshold_for_trafo_bus_creation (Optional[float]): If the voltage level of transformer locations
            is far different than the transformer data, additional buses are created.
            rel_deviation_threshold_for_trafo_bus_creation defines the tolerance in which no additional buses are
            created. (default: 0.2)
        log_rel_vn_deviation (Optional[float]): This parameter allows a range below
            rel_deviation_threshold_for_trafo_bus_creation in which a warning is logged instead of a creating additional
            buses. (default: 0.12)

    Returns:
        net created from the jao data

    Example:
        >>> from pathlib import Path
        >>> import os
        >>> from pandapower.converter.jao.from_jao import from_jao
        >>> net = from_jao()
        >>> home = str(Path.home())
        >>> # assume that the files are located at your desktop:
        >>> excel_file_path = os.path.join(home, "desktop", "202409_Core Static Grid Mode_6th release")
        >>> html_file_path = os.path.join(home, "desktop", "2024-09-13_Core_SGM_publication.html")
        >>> net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
    """

    # --- read data
    data = pd.read_excel(excel_file_path, sheet_name=None, header=[0, 1])
    if html_file_path is not None:
        with open(html_file_path, mode="r", encoding=kwargs.get("encoding", "utf-8")) as f:
            html_str = f.read()
    else:
        html_str = ""

    # --- manipulate data / data corrections
    if apply_data_correction:
        html_str = _data_correction(data, html_str, max_i_ka_fillna)  # type: ignore[assignment]

    # --- parse html_str to line_geo_data
    line_geo_data = None
    if html_str:
        try:
            line_geo_data = _parse_html_str(html_str)
        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            logger.error(f"html data were ignored due to this error:\n{e}")

    # --- create the pandapower net
    net = pandapowerNet(name=os.path.splitext(os.path.basename(excel_file_path))[0], sn_mva=kwargs.get("sn_mva", 1.0))
    _create_buses_from_line_data(net, data)
    _create_lines(net, data, max_i_ka_fillna)
    _create_transformers_and_buses(net, data, **kwargs)

    # --- invent connections between grid groups
    if extend_data_for_grid_group_connections:
        _invent_connections_between_grid_groups(net, **kwargs)

    # --- drop islanded grid groups
    if drop_grid_groups_islands:
        drop_islanded_grid_groups(net, kwargs.get("min_bus_number", 6))

    # --- add geo data to buses and lines
    if line_geo_data is not None:
        _add_bus_geo(net, line_geo_data)
        set_line_geodata_from_bus_geodata(net)

    return net


if __name__ == "__main__":
    from pathlib import Path

    from pandapower.converter.jao._geodata import _fill_geo_at_one_sided_branches_without_geo_extent
    from pandapower.converter.jao._grid_groups import get_grid_groups
    from pandapower.file_io import from_json, to_json

    home = str(Path.home())
    jao_data_folder = os.path.join(home, "Documents", "JAO Static Grid Model")

    release5 = os.path.join(jao_data_folder, "20240329_Core Static Grid Model – 5th release")
    excel_file_path = os.path.join(release5, "20240329_Core Static Grid Model_public.xlsx")
    html_file_path = os.path.join(
        release5, "20240329_Core Static Grid Model Map_public", "2024-03-18_Core_SGM_publication.html"
    )

    release6 = os.path.join(jao_data_folder, "202409_Core Static Grid Mode_6th release")
    excel_file_path = os.path.join(release6, "20240916_Core Static Grid Model_for publication.xlsx")
    html_file_path = os.path.join(
        release6, "2024-09-13_Core_SGM_publication_files", "2024-09-13_Core_SGM_publication.html"
    )

    pp_net_json_file = os.path.join(home, "desktop", "jao_grid.json")

    if 1:  # read from original data
        net = from_jao(excel_file_path, html_file_path, True, drop_grid_groups_islands=True)
        to_json(net, pp_net_json_file)
    else:  # load net from already converted and stored net
        net = from_json(pp_net_json_file)

    print(net)
    grid_groups = get_grid_groups(net)
    print(grid_groups)

    _fill_geo_at_one_sided_branches_without_geo_extent(net)
