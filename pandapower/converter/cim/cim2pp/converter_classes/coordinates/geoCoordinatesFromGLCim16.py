import logging
import time

import numpy as np
import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.geoCoordinatesFromGLCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class GeoCoordinatesFromGLCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def add_geo_coordinates_from_gl_cim16(self):
        self.logger.info("Creating the geo coordinates from CGMES GeographicalLocation.")
        time_start = time.time()
        gl_data = pd.merge(
            self.cimConverter.cim['gl']['PositionPoint'][['Location', 'xPosition', 'yPosition', 'sequenceNumber']],
            self.cimConverter.cim['gl']['Location'][['rdfId', 'PowerSystemResources']], how='left',
            left_on='Location', right_on='rdfId')
        gl_data = gl_data.drop(columns=['Location', 'rdfId'])
        # make sure that the columns 'xPosition' and 'yPosition' are floats
        gl_data['xPosition'] = gl_data['xPosition'].astype(float)
        gl_data['yPosition'] = gl_data['yPosition'].astype(float)
        bus_geo = gl_data.rename(columns={'PowerSystemResources': 'Substation'})
        cn = self.cimConverter.cim['eq']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]
        cn = pd.concat([cn, self.cimConverter.cim['eq_bd']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cimConverter.cim['tp']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cimConverter.cim['tp_bd']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = cn.rename(columns={'rdfId': sc['o_id'], 'ConnectivityNodeContainer': 'rdfId'})
        cn = pd.merge(cn, self.cimConverter.cim['eq']['VoltageLevel'][['rdfId', 'Substation']], how='left', on='rdfId')
        cn = cn.drop(columns=['rdfId'])
        buses = self.cimConverter.net.bus.reset_index()
        buses = buses[['index', sc['o_id']]]
        buses = pd.merge(buses, cn, how='left', on=sc['o_id'])
        bus_geo = pd.merge(bus_geo, buses, how='inner', on='Substation')
        bus_geo = bus_geo.drop(columns=['Substation'])
        bus_geo = bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'])
        bus_geo['coords'] = bus_geo[['xPosition', 'yPosition']].values.tolist()
        bus_geo['coords'] = bus_geo[['coords']].values.tolist()
        # for the buses which have more than one coordinate
        bus_geo_mult = bus_geo[bus_geo.duplicated(subset=sc['o_id'], keep=False)]
        # now deal with the buses which have more than one coordinate
        for _, df_group in bus_geo_mult.groupby(by=sc['o_id'], sort=False):
            bus_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        bus_geo = bus_geo.drop_duplicates([sc['o_id']], keep='first')
        bus_geo = bus_geo.sort_values(by='index')
        start_index_pp_net = self.cimConverter.net.bus_geodata.index.size
        self.cimConverter.net.bus_geodata = pd.concat(
            [self.cimConverter.net.bus_geodata, pd.DataFrame(None, index=bus_geo['index'].values)],
            ignore_index=False, sort=False)
        self.cimConverter.net.bus_geodata.x[start_index_pp_net:] = bus_geo.xPosition[:]
        self.cimConverter.net.bus_geodata.y[start_index_pp_net:] = bus_geo.yPosition[:]
        self.cimConverter.net.bus_geodata.coords[start_index_pp_net:] = bus_geo.coords[:]
        # reduce to max two coordinates for buses (see pandapower documentation for details)
        self.cimConverter.net.bus_geodata['coords_length'] = self.cimConverter.net.bus_geodata['coords'].apply(len)
        self.cimConverter.net.bus_geodata.loc[
            self.cimConverter.net.bus_geodata['coords_length'] == 1, 'coords'] = np.nan
        self.cimConverter.net.bus_geodata['coords'] = self.cimConverter.net.bus_geodata.apply(
            lambda row: [row['coords'][0], row['coords'][-1]] if row['coords_length'] > 2 else row['coords'], axis=1)
        if 'coords_length' in self.cimConverter.net.bus_geodata.columns:
            self.cimConverter.net.bus_geodata = self.cimConverter.net.bus_geodata.drop(columns=['coords_length'])

        # the geo coordinates for the lines
        lines = self.cimConverter.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = gl_data.rename(columns={'PowerSystemResources': sc['o_id']})
        line_geo = pd.merge(line_geo, lines, how='inner', on=sc['o_id'])
        line_geo = line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'])
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for _, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo = line_geo.drop_duplicates([sc['o_id']], keep='first')
        line_geo = line_geo.sort_values(by='index')
        # now add the line coordinates
        start_index_pp_net = self.cimConverter.net.line_geodata.index.size
        self.cimConverter.net.line_geodata = pd.concat(
            [self.cimConverter.net.line_geodata, pd.DataFrame(None, index=line_geo['index'].values)],
            ignore_index=False, sort=False)
        self.cimConverter.net.line_geodata.coords[start_index_pp_net:] = line_geo.coords[:]

        # now create geo coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'impedance', 'dcline', 'shunt',
                        'storage', 'ward', 'xward']:
            one_ele_df = self.cimConverter.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(gl_data.rename(columns={'PowerSystemResources': sc['o_id']}),
                                  one_ele_df, how='inner', on=sc['o_id'])
            one_ele_df = one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'])
            one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
            one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
            for _, df_group in one_ele_df.groupby(by=sc['o_id']):
                one_ele_df['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            one_ele_df = one_ele_df.drop_duplicates([sc['o_id']], keep='first')
            # now add the coordinates
            self.cimConverter.net[one_ele]['coords'] = self.cimConverter.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('coords'))

        self.logger.info("Finished creating the GL coordinates, needed time: %ss" % (time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the GL coordinates, needed time: %ss" % (time.time() - time_start)))
