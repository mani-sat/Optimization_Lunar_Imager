import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import os


class DL_optimizer_gw:
    def __init__(self, M, buffersize):
        self.M = M
        self.buffersize = buffersize
    
    def setup(self, Rsc:float, Rdl:float):
        self.Rsc = Rsc
        self.Rdl = Rdl
    
    def load_data(self, folder_path):
        with open(f"{folder_path}/som.pickle", 'rb') as f:
            full_som = pickle.load(f)
        f.close()
        with open(f"{folder_path}/los.pickle", "rb") as f:
            full_los= pickle.load(f)
        f.close()
        with open(f"{folder_path}/rs.pickle", "rb") as f:
            Rsc_init = pickle.load(f)
        f.close()
        with open(f"{folder_path}/rd.pickle", "rb") as f:
            Rdl_init = pickle.load(f)
        f.close()
        with open(f"{folder_path}/outage_los.pickle", "rb") as f:
            outage_los = pickle.load(f)
        f.close()
        Rsc=Rsc_init #factor science rate up
        Rdl=Rdl_init
        return full_som, full_los, Rsc, Rdl, outage_los

    def split_data(self, full_los):
        total_len=len(full_los)
        length=total_len/12
        indexes = [int(length*i) for i in range(13)]
        return indexes

    def get_data(self, full_sos, full_los, indexes:list):
        """
        indexes: length 2 list of start index and end index
        """
        T_sos=full_sos[indexes[0]:indexes[1]]
        T_los=full_los[indexes[0]:indexes[1]]
        return T_sos, T_los

    def optimize(self, som, los, buffer):
        buffer_size = self.buffersize
        sun_light = som
        los_sc = los
        N=len(som)

        Rsc = self.Rsc
        Rdl = self.Rdl

        # Decision variables
        T_sc = cp.Variable(N, boolean=True)
        T_dl = cp.Variable(N, boolean=True)
        T_idle = cp.Variable(N, boolean=True)

        # Constraints list
        dl_cumsum = cp.cumsum(T_dl) * Rdl
        sc_cumsum = cp.cumsum(T_sc) * Rsc

        # Constraints list
        constraints = []

        # Constraint 1: Must only perform action when in state
        constraints += [T_sc <= sun_light]
        constraints += [T_dl <= los_sc]

        # Constraint 2: Only one variable avaiable at each time step
        constraints += [T_sc + T_dl + T_idle == 1]

        # Constraint 3: We cannot transmit what we have not scienced
        constraints += [dl_cumsum <= sc_cumsum + buffer]

        # Constraint 4: Must not exceed buffer size
        constraints += [sc_cumsum - dl_cumsum <= buffer_size]
        constraints += [sc_cumsum - dl_cumsum >= 0]

        # Def of totals
        sc_total = cp.sum(T_sc)*Rsc
        dl_total = cp.sum(T_dl)*Rdl
            
        # Objective: minimize total downlink usage
        objective = cp.Maximize(dl_total)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        print(cp.installed_solvers())

        # Ensure the CBC solver is installed
        problem.solve(solver=cp.GUROBI, verbose=False, save_file="model.lp", TimeLimit=7200)
        # Output
        print("Status:", problem.status)
        print("Total cost (timeslots slots used):", problem.value)
        print(f"Total downlinked: {dl_total.value / 1e9:.5f} Mbit")
        print(f"Total scienced: {sc_total.value / 1e9:.5f} Mbit")

        # return T_sc.value, T_dl.value, T_idle.value, T_gs.value
        return T_sc.value, T_dl.value, T_idle.value
    
def log(msg):
    print(f"[{datetime.datetime.now()}] {msg}")


if __name__=="__main__":
    log("starting...")
    test_id = 33
    # data_folder = 'station_AAU_rate_59677090'
    data_folder = 'station_NN11_rate_404820636'
    station = data_folder.split("_")[1]

    M=60
    if station=="AAU":
        M=30
    buffersize=250e9*8


    test_name = 'final_results/Optim_results_raw'
    save_folder=os.path.join("./Optimization", test_name)
    save_folder=os.path.join(save_folder, station)
    os.makedirs(save_folder, exist_ok=True)
    optimizer=DL_optimizer_gw(M, buffersize)
    #load data and get rates
    full_som, full_los, Rsc, Rdl, outage_los = optimizer.load_data("Optimization/"+data_folder)
    log(f"Rsc:{Rsc}, Rdl:{Rdl}")
    indexes=optimizer.split_data(full_los)
    log(indexes)
    
    #setup rates
    with open(f'./{save_folder}/Rsc.pickle', 'wb') as f:
        pickle.dump(Rsc, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/Rdl.pickle', 'wb') as f:
        pickle.dump(Rdl, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/outage_los.pickle', 'wb') as f:
        pickle.dump(outage_los, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    optimizer.setup(Rsc, Rdl)

    month=0
    log(f"month process initiating: {month}")
    buffer=0
    T_som, T_los=optimizer.get_data(full_som, full_los, [indexes[month], indexes[month+1]])
    log(f"Month length in samples: {len(T_som)}")
    log(f"Data aquired")
    T_sc, T_dl, T_idle = optimizer.optimize(T_som, T_los, buffer)
    log(f"Done optimizing for month:{month}")
    
    # plt.savefig(f"{save_folder}/test111.png")
    with open(f'./{save_folder}/T_sc.pickle', 'wb') as f:
        pickle.dump(T_sc, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/T_dl.pickle', 'wb') as f:
        pickle.dump(T_dl, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./{save_folder}/T_idle.pickle', 'wb') as f:
        pickle.dump(T_idle, f, protocol=pickle.HIGHEST_PROTOCOL)

    sc_cumsum = (np.cumsum(T_sc)*Rsc+buffer)/1e9
    dl_cumsum = np.cumsum(T_dl)*Rdl/1e9

    plt.rcParams.update({'font.size': 11})
    plt.step(range(len(T_dl)), sc_cumsum, label = 'Accumulated science', color="tab:orange")
    plt.step(range(len(T_dl)), dl_cumsum, label = 'Accumulated GS downlink', color="tab:blue")
    # plot_rectagles(outages, label='Outages')
    plt.step(range(len(T_dl)), sc_cumsum + buffer - dl_cumsum, label = 'buffer', color="tab:red")
    plt.axhline(buffersize/1e9, c='tab:red', linestyle='--', label='Max buffer size', color="tab:red")
    plt.legend(loc="upper left")
    plt.xticks(
        ticks=np.arange(0, len(T_dl), 2*24*60),
        labels=[str(int(i/(24*60))) for i in np.arange(0, len(T_dl), 2*24*60)]
    )
    plt.xlabel("Time [days]")
    # plt.xlabel("Time [min]")
    plt.xlim(0, len(T_dl))
    plt.ylabel("Acculmulated Data [GB]")
    plt.title(f"Accumulated data over one month for {station}")
    extra_folder="figs"
    os.makedirs(extra_folder, exist_ok=True)
    plt.savefig(f"{extra_folder}/raw_result_{station}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    log("Done.")