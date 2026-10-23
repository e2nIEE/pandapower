__all__ = [
    "draw_layers",
    "version_check",
    "get_plotly_cmap",
    "get_plotly_color",
    "get_plotly_color_palette",
    "layers_plotly",
    "pf_res_plotly",
    "get_hoverinfo",
    "simple_plotly",
    "create_bus_trace",
    "create_dcline_trace",
    "create_edge_center_trace",
    "create_line_trace",
    "create_scale_trace",
    "create_weighted_marker_trace",
    "draw_traces",
    "get_line_neutral",
    "sum_line_length",
    "vlevel_plotly",
    "add_cmap_to_collection",
    "add_collections_to_axes",
    "add_single_collection",
    "create_annotation_collection",
    "create_bus_bus_switch_collection",
    "create_bus_collection",
    "create_busbar_collection",
    "create_dcline_collection",
    "create_ext_grid_collection",
    "create_gen_collection",
    "create_impedance_collection",
    "create_line_collection",
    "create_line_switch_collection",
    "create_load_collection",
    "create_sgen_collection",
    "create_storage_collection",
    "create_trafo3w_collection",
    "create_trafo3w_connection_collection",
    "create_trafo_collection",
    "create_trafo_connection_collection",
    "create_vsc_collection",
    "create_vsc_connection_collection",
    "create_ward_collection",
    "create_xward_collection",
    "CustomTextPath",
    "draw_collections",
    "TextPath",
    "cmap_continuous",
    "cmap_discrete",
    "cmap_logarithmic",
    "build_igraph_from_pp",
    "coords_from_igraph",
    "coords_from_nxgraph",
    "create_generic_coordinates",
    "fuse_geodata",
    "convert_crs",
    "convert_geodata_to_geojson",
    "convert_gis_to_geojson",
    "dump_to_geojson",
    "dump_to_geojson_node_branch",
    "set_line_geodata_from_bus_geodata",
    "get_collection_sizes",
    "plot_loading",
    "plot_voltage_profile",
    "voltage_profile_to_bus_geodata",
    "bus_info",
    "calculate_unique_angles",
    "hover",
    "line_info",
    "simple_plot",
    "trafo3w_info",
    "trafo_info",
    "to_html",
]

import types

from pandapower.plotting.collections import (
    add_cmap_to_collection,
    add_collections_to_axes,
    add_single_collection,
    create_annotation_collection,
    create_bus_bus_switch_collection,
    create_bus_collection,
    create_busbar_collection,
    create_dcline_collection,
    create_ext_grid_collection,
    create_gen_collection,
    create_impedance_collection,
    create_line_collection,
    create_line_switch_collection,
    create_load_collection,
    create_sgen_collection,
    create_storage_collection,
    create_trafo3w_collection,
    create_trafo3w_connection_collection,
    create_trafo_collection,
    create_trafo_connection_collection,
    create_vsc_collection,
    create_vsc_connection_collection,
    create_ward_collection,
    create_xward_collection,
    CustomTextPath,
    draw_collections,
    TextPath,
)
from pandapower.plotting.colormaps import cmap_continuous, cmap_discrete, cmap_logarithmic
from pandapower.plotting.generic_geodata import (
    build_igraph_from_pp,
    coords_from_igraph,
    coords_from_nxgraph,
    create_generic_coordinates,
    fuse_geodata,
)
from pandapower.plotting.geo import (
    convert_crs,
    convert_geodata_to_geojson,
    convert_gis_to_geojson,
    dump_to_geojson,
    dump_to_geojson_node_branch,
)
from pandapower.plotting.plotly import (
    draw_layers,
    version_check,
    get_plotly_cmap,
    get_plotly_color,
    get_plotly_color_palette,
    layers_plotly,
    pf_res_plotly,
    get_hoverinfo,
    simple_plotly,
    create_bus_trace,
    create_dcline_trace,
    create_edge_center_trace,
    create_line_trace,
    create_scale_trace,
    create_weighted_marker_trace,
    draw_traces,
    get_line_neutral,
    sum_line_length,
    vlevel_plotly,
)
from pandapower.plotting.plotting_toolbox import set_line_geodata_from_bus_geodata, get_collection_sizes
from pandapower.plotting.powerflow_results import plot_loading, plot_voltage_profile, voltage_profile_to_bus_geodata
from pandapower.plotting.simple_plot import (
    bus_info,
    calculate_unique_angles,
    hover,
    line_info,
    simple_plot,
    trafo3w_info,
    trafo_info,
)
from pandapower.plotting.to_html import to_html

try:
    from matplotlib.backend_bases import GraphicsContextBase, RendererBase

    class GC(GraphicsContextBase):
        def __init__(self):
            super().__init__()
            self.set_capstyle("round")

    def custom_new_gc(self):
        return GC()

    RendererBase.new_gc = types.MethodType(custom_new_gc, RendererBase)  # type: ignore[method-assign]

except ImportError:
    pass
