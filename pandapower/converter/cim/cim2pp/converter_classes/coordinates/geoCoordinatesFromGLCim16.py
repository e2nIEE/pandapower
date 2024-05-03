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
        gl_data['coords_str'] = gl_data['xPosition'].astype(str) + ', ' + gl_data['yPosition'].astype(str)
        # deal with nodes: the geo data is not directly attached to the node, they are attached to the substation
        bus_geo = gl_data.rename(columns={'PowerSystemResources': 'Substation'})
        cn = self.cimConverter.cim['eq']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]
        cn = pd.concat([cn, self.cimConverter.cim['eq_bd']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cimConverter.cim['tp']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        if 'tp_bd' in self.cimConverter.cim.keys():  # check because tp_bd was removed in cgmes 3.0
            cn = pd.concat([cn, self.cimConverter.cim['tp_bd']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = cn.rename(columns={'rdfId': sc['o_id'], 'ConnectivityNodeContainer': 'rdfId'})
        cn = pd.merge(cn, self.cimConverter.cim['eq']['VoltageLevel'][['rdfId', 'Substation']], how='left', on='rdfId')
        cn = cn.drop(columns=['rdfId'])
        buses = pd.merge(self.cimConverter.net.bus[[sc['o_id']]], cn, how='left', on=sc['o_id'])
        bus_geo = pd.merge(bus_geo, buses, how='inner', on='Substation')
        bus_geo.drop(columns=['Substation'], inplace=True)
        bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        # only one coordinate for each asset (except line and impedance)
        bus_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        bus_geo['geo'] = '{"coordinates": [' + bus_geo['coords_str'] + '], "type": "Point"}'
        self.cimConverter.net['bus']['geo'] = self.cimConverter.net['bus'][sc['o_id']].map(
            bus_geo.set_index(sc['o_id']).to_dict(orient='dict').get('geo'))

        # the geo coordinates for the lines
        lines = self.cimConverter.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = gl_data.rename(columns={'PowerSystemResources': sc['o_id']})
        line_geo = pd.merge(line_geo, lines, how='inner', on=sc['o_id'])
        line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for _, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        line_geo.sort_values(by='index', inplace=True)
        line_geo['geo'] = '{"coordinates": ' + line_geo['coords'].astype(str) + ', "type": "LineString"}'
        # now add the line coordinates
        self.cimConverter.net['line']['geo'] = self.cimConverter.net['line'][sc['o_id']].map(
            line_geo.set_index(sc['o_id']).to_dict(orient='dict').get('geo'))

        gl_data = gl_data.rename(columns={'PowerSystemResources': sc['o_id']})
        # now create geo coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'dcline', 'impedance',
                        'shunt', 'storage', 'ward', 'xward']:
            one_ele_df = self.cimConverter.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(gl_data, one_ele_df, how='inner', on=sc['o_id'])
            one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
            if one_ele not in ['dcline', 'impedance']:
                # only one coordinate for each asset (except line and impedance)
                one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
                one_ele_df['geo'] = '{"coordinates": [' + one_ele_df["coords_str"] + '], "type": "Point"}'
            else:
                # line strings
                one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
                one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
                for _, df_group in one_ele_df.groupby(by=sc['o_id']):
                    one_ele_df['coords'][df_group.index.values[0]] = df_group[
                        ['xPosition', 'yPosition']].values.tolist()
                one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
                one_ele_df['coords'] = one_ele_df['coords'].astype(str)
                one_ele_df['geo'] = '{"coordinates": ' + one_ele_df['coords'].astype(str) + ', "type": "LineString"}'
            # now add the coordinates
            self.cimConverter.net[one_ele]['geo'] = self.cimConverter.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('geo'))

        self.logger.info("Finished creating the GL coordinates, needed time: %ss" % (time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the GL coordinates, needed time: %ss" % (time.time() - time_start)))
