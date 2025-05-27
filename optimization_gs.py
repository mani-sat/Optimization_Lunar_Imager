import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import os

def plot_rectagles(lst, color='tab:blue', alpha=0.3, label=None):
    in_region = False
    start = 0
    label_added = False  # Track if label has been added

    for i, val in enumerate(lst + [False]):
        if val and not in_region:
            start = i
            in_region = True
        elif not val and in_region:
            if label and not label_added:
                plt.axvspan(start-1, i-1, color=color, alpha=alpha, label=label)
                label_added = True
            else:
                plt.axvspan(start-1, i-1, color=color, alpha=alpha)
            in_region = False

class DL_optimizer_gw:
    def __init__(self, M, buffersize):
        self.M = M
        self.buffersize = buffersize
    
    def setup(self, Rsc:float, Rdl:float, alpha, rho, c1):
        self.Rsc = Rsc
        self.Rdl = Rdl
        self.alpha = alpha
        self.rho = rho
        self.c1 = c1
    
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
        M = self.M
        buffer_size = self.buffersize
        sun_light = som
        los_sc = los
        N=len(som)
        c1 = self.c1

        rho = self.rho
        Rsc = self.Rsc
        Rdl = self.Rdl
        # Decision variables
        T_sc = cp.Variable(N, boolean=True)
        T_dl = cp.Variable(N, boolean=True)
        T_idle = cp.Variable(N, boolean=True)

        # Constraints list
        dl_cumsum = cp.cumsum(T_dl)*Rdl
        sc_cumsum = cp.cumsum(T_sc)*Rsc

        # Ground station cappin
        rising_edges = cp.Variable(N, boolean=True)
        falling_edges = cp.Variable(N, boolean=True)
        re_cumsum = cp.Variable(N)
        fe_cumsum = cp.Variable(N)
        T_gs = cp.Variable(N, boolean=True)

        # Constraints list
        constraints = []

        # Def of totals
        sc_total = cp.sum(T_sc)*Rsc
        dl_total = cp.sum(T_dl)*Rdl
        gs_total = cp.sum(T_gs)*Rdl

        # Constraint 1: Must only perform action when in state
        constraints += [T_sc <= sun_light]
        constraints += [T_dl <= los_sc]

        # Constraint 2: Only one variable avaiable at each time step
        constraints += [T_sc + T_dl + T_idle  == 1]

        # Constraint 3: We cannot transmit what we have not scienced
        constraints += [dl_cumsum <= sc_cumsum + buffer]

        # Constraint 4: Must not exceed buffer size
        constraints += [sc_cumsum + buffer - dl_cumsum  <= buffer_size]
        constraints += [sc_cumsum + buffer - dl_cumsum  >= 0]

        constraints += [falling_edges[:M] == 0]

        constraints += [falling_edges[M:N] == rising_edges[:(N-M)]]
        constraints += [(cp.sum(rising_edges[:N-M]) - cp.sum(falling_edges[:N-M])) >= 0]
        constraints += [(cp.sum(rising_edges[:N-M]) - cp.sum(falling_edges[:N-M])) <= 1]

        constraints += [rising_edges[N-M:N] == 0]
        constraints += [T_gs == (re_cumsum - fe_cumsum)]
        constraints += [T_dl <= T_gs]
            
        # Objective: minimize total downlink usage
        objective = cp.Maximize((1 / c1) * (dl_total - rho * gs_total))
        
        # Solve
        problem = cp.Problem(objective, constraints)
        print(cp.installed_solvers())

        problem.solve(solver=cp.GUROBI, verbose=False, TimeLimit=7200)

        # Output
        print("Status:", problem.status)
        print("Total cost (timeslots slots used):", problem.value)
        print(f"Total downlined: {dl_total.value / 1e9:.5f} Mbit")
        print(f"Total scienced: {sc_total.value / 1e9:.5f} Mbit")

        return T_sc.value, T_dl.value, T_idle.value, T_gs.value
    
    def analyze(self, folder_path, T_dl, R_dl, T_gs, cost_gs, rho):
        # need outage_los, Rdl_raw
        tot_dl=sum(T_dl)*R_dl
        with open(f"{folder_path}/outage_los.pickle", 'rb') as f:
            outage_los = pickle.load(f)
        f.close()
        with open(f"{folder_path}/rd_raw.pickle", 'rb') as f:
            rd_raw = pickle.load(f)
        f.close()
        T_dl=T_dl.astype(bool)
        outages=T_dl&(outage_los[:len(T_dl)]==False)

        time_with_outages=T_dl&outage_los[:len(T_dl)]
        tot_dl_outages=sum(time_with_outages)*rd_raw
        end_buffer = tot_dl-tot_dl_outages
        total_DL=sum(T_dl)*R_dl
        # total_cost=(sum(T_gs)*R_dl)*cost_gs
        total_cost=(sum(T_gs))*cost_gs
        return outages, end_buffer, total_cost, total_DL
    
    def plotting(self, rho, station, T_dl, T_sc, T_gs, Rdl, Rsc, outages, buffer_size, save_folder, buffer=0):
        buffer_size = buffer_size/1e9
        sc_cumsum = np.cumsum(T_sc)*Rsc/1e9
        dl_cumsum = np.cumsum(T_dl)*Rdl/1e9

        plt.rcParams.update({'font.size': 15})
        plt.step(range(len(T_dl)), sc_cumsum, label = 'Accumulated science', color="tab:orange")
        plt.step(range(len(T_dl)), dl_cumsum, label = 'Accumulated GS downlink', color="tab:blue")
        # plot_rectagles(outages, label='Outages')
        plt.step(range(len(T_dl)), sc_cumsum + buffer - dl_cumsum, label = 'buffer', color="tab:red")
        plt.axhline(buffer_size, c='tab:red', linestyle='--', label='Max buffer size', color="tab:red")
        plt.legend(loc="upper left")
        plt.xticks(
            ticks=np.arange(0, len(T_dl), 2*24*60),
            labels=[str(int(i/(24*60))) for i in np.arange(0, len(T_dl), 2*24*60)]
        )
        plt.xlabel("Time [days]")
        # plt.xlabel("Time [min]")
        plt.xlim(0, len(T_dl))
        plt.ylabel("Acculmulated Data [Gb]")
        # plt.title(fr"Accumulated data over one month ($\rho$={rho:.2f}).")
        plt.savefig(f"./{save_folder}/gs_result_{station}_{rho:.2f}.pdf", format="pdf", bbox_inches="tight")
        #an extra save for ease
        extra_folder="figs"
        os.makedirs(extra_folder, exist_ok=True)
        plt.savefig(f"./{extra_folder}/gs_result_{station}_{rho:.2f}.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        



def log(msg):
    print(f"[{datetime.datetime.now()}] {msg}")

if __name__=="__main__":
    log("starting...")
    #Parameters:

    test_id = 0
    data_folders = ['station_NN11_rate_404820636', 'station_AAU_rate_59677090']
    # data_folders = ['station_AAU_rate_59677090']
    # data_folders = ['station_NN11_rate_404820636']
    for data_folder in data_folders:
        station = data_folder.split("_")[1]
        M=60
        c1 = 1
        if station=="AAU":
            M=30
            c1=0.1
        buffersize=250e9*8
        alpha=1
        rho_list =[1/5, 1/3, 1/2, 2/3]
        buffer=0
        

        test_name = 'Optim_results_gs'
        save_folder=os.path.join("./Optimization/final_results", test_name)
        save_folder=os.path.join(save_folder, station)
        os.makedirs(save_folder, exist_ok=True)
        optimizer=DL_optimizer_gw(M, buffersize)
        #load data and get rates
        full_som, full_los, Rsc, Rdl, outage_los=optimizer.load_data("Optimization/"+data_folder)
        log(f"Rsc:{Rsc}, Rdl:{Rdl}")
        indexes=optimizer.split_data(full_los)
        log(indexes)
        for rho in rho_list:
            log(f"Starting rho: {rho}")

            test_folder=os.path.join(save_folder, f"util_{rho:.2f}")
            os.makedirs(test_folder,exist_ok=True)
            with open(f'./{test_folder}/Rsc.pickle', 'wb') as f:
                pickle.dump(Rsc, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'./{test_folder}/Rdl.pickle', 'wb') as f:
                pickle.dump(Rdl, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'./{test_folder}/outage_los.pickle', 'wb') as f:
                pickle.dump(outage_los, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            optimizer.setup(Rsc, Rdl, alpha, rho, c1)
            # for i in range(len(indexes)-1):

            month=0
            log(f"rho process initiating: {rho}")
            T_som, T_los=optimizer.get_data(full_som, full_los, [indexes[month], indexes[month+1]])
            # log(f"rho length in samples: {len(rho)}")
            log(f"Data aquired")
            T_sc, T_dl, T_idle, GS=optimizer.optimize(T_som, T_los, buffer)
            log(f"Done optimizing for rho:{rho}")

            outages, end_buffer, total_cost, total_DL=optimizer.analyze("Optimization/"+data_folder, T_dl, Rdl, GS, c1, rho)
            # extra_folder="final_results/Optim_results_gs"
            extra_folder="Processing"
            with open(f'./{extra_folder}/results_summary_gs.txt', "a") as summary_file_eng:
                summary_file_eng.write(f"{rho:.2f} & {station} & {total_cost:.3e} & {(end_buffer/1e9):.2f}GB & {(total_DL/1e9):.2f} \\\\\n")
            with open(f'./{test_folder}/end_buffer.txt', 'w') as f:
                f.write(f"data in buffer end (outages): {end_buffer}\n")
                f.write(f"total cost: {total_cost}")
            f.close()
            log(f"Analysis done. end buffer: {end_buffer}")

            optimizer.plotting(rho, station, T_dl, T_sc, GS, Rdl, Rsc, outages, buffersize, test_folder)
            log("Plotting done")
            # plt.savefig(f"{save_folder}/test111.png")
            with open(f'./{test_folder}/T_sc.pickle', 'wb') as f:
                pickle.dump(T_sc, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'./{test_folder}/T_dl.pickle', 'wb') as f:
                pickle.dump(T_dl, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'./{test_folder}/T_idle.pickle', 'wb') as f:
                pickle.dump(T_idle, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f'./{test_folder}/GS.pickle', 'wb') as f:
                pickle.dump(GS, f, protocol=pickle.HIGHEST_PROTOCOL)
            log(f"Saved to pickle, rho: {rho:.2f}")
    log("Done for gs")