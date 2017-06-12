import pandas as pd
from pandapower.create import create_empty_network


def from_sql(netname, engine, include_empty_tables=False, include_results=True):

    with engine.connect() as conn, conn.begin():
        net = create_empty_network()
        for item, table in net.items():
            if not isinstance(table,pd.DataFrame) or item.startswith("_"):
                continue
            elif item.startswith("res"):
                if include_results and engine.has_table(netname + "_" + item):
                    net[item] = pd.read_sql_table(netname + "_" + item, conn)
           # elif item == "line_geodata":
            #    geo = pd.DataFrame(index=table.index)
             #   for i, coord in table.iterrows():
           #         for nr, (x, y) in enumerate(coord.coords):
           #             geo.loc[i, "x%u" % nr] = x
           #             geo.loc[i, "y%u" % nr] = y
           #     geo.to_sql(net.name+"_"+item,engine)

            elif engine.has_table(netname + "_" + item) or include_empty_tables:
                net[item] = pd.read_sql_table(netname + "_" + item, conn)

        para = pd.read_sql_table(netname + "_" + "parameters", conn)
        net.name, net.f_hz, net.version = para['parameters'][0], para['parameters'][1], para["parameters"][2]
        net.std_types["line"] = pd.read_sql_table('line_std_types', conn)
        net.std_types["trafo"] = pd.read_sql_table("trafo_std_types", conn)
        net.std_types["trafo3w"] = pd.read_sql_table("trafo3w_std_types", conn)
        return net


def to_sql(net, engine, include_empty_tables=False, include_results=True):

    for item, table in net.items():
        if not isinstance(table,pd.DataFrame) or item.startswith("_"):
            continue
        elif item.startswith("res"):
            if include_results and len(table) > 0:
                table.to_sql(net.name + "_" + item,
                             engine, if_exists="replace")
        elif item == "line_geodata":
            geo = pd.DataFrame(index=table.index)
            for i, coord in table.iterrows():
                for nr, (x, y) in enumerate(coord.coords):
                    geo.loc[i, "x%u" % nr] = x
                    geo.loc[i, "y%u" % nr] = y
            geo.to_sql(net.name + "_" + item, engine, if_exists="replace")
        elif len(table) > 0 or include_empty_tables:
            table.to_sql(net.name + "_" + item, engine, if_exists="replace")
    parameters = pd.DataFrame(index=["name", "f_hz", "version"], columns=["parameters"],
                              data=[net.name, net.f_hz, net.version])
    pd.DataFrame(net.std_types["line"]).T.to_sql('line_std_types', engine, if_exists='append')
    pd.DataFrame(net.std_types["trafo"]).T.to_sql("trafo_std_types", engine, if_exists='append')
    pd.DataFrame(net.std_types["trafo3w"]).T.to_sql("trafo3w_std_types", engine, if_exists='append')
    parameters.to_sql(net.name + "_" + "parameters",engine, if_exists="replace")
