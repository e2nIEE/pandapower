from fastapi import FastAPI
import uvicorn
from test_sim import net, run_simulation

app = FastAPI()

@app.get("/grid-power-analysis/")
def grid_power_analysis(): 
    # run network simulation
    run_simulation()

    # create a function to convert series to dictionary
    def convert_to_dict(series):
        return series.to_dict()

    # convert res_load and load name table to dictionary
    load = convert_to_dict(net.res_load)
    load_name = convert_to_dict(net.load.name)

    # get active and reactive load power and put to a list.
    load_power = [load["p_mw"][0],load["q_mvar"][0]]

    # add load node data to dictionary.
    node = dict(loads = {load_name[0] : tuple(load_power)})

    
    # convert external_grid and external_grid name table to dictionary
    external_grid = convert_to_dict(net.res_ext_grid)
    external_grid_name = convert_to_dict(net.ext_grid.name)

    # get active and reactive external_grid power and put to a list.
    external_grid_power = [external_grid["p_mw"][0], external_grid["q_mvar"][0]] 

    # add external_grid_power data to dictionary.
    external_grid_node = dict(external_grid_connection = {external_grid_name[0] : tuple(external_grid_power)})

    # add external grid node to dictionary
    node.update(external_grid_node)

    return node

if __name__ == '__main__':   
    uvicorn.run('main:app', reload=True)



