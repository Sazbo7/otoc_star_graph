import sys
from star_graph import *
import pandas as pd

### Data saved using Pandas DataFrame to .pickle files ####
df_columns = ["L", "t_max", "t_init", "t_step", "Jzz", "Jx", "Jz", "Boundary", "Seed","Coupling", "OTOC_t"];
df = pd.DataFrame(columns=df_columns);


coupling_vals = np.linspace(0, 3, 13);
length_vals = np.arange(10, 18, 1);

count = 0;
t_max, t_step, t_init, seed, periodic = 15, 400, 0, None, False

for L in length_vals:
    t_max, t_init, values = run_OTOC(-1.0, 0.0, 1.05, coupling_vals, initial_state="Haar", t_max = t_max, t_step=t_step, t_init=t_init, seed=seed, L=L, periodic=periodic);

    df = df.append({"L": L,"t_max":, "t_init":t_init, "t_step":t_step, "Jzz":-1.0, "Jx":1.05, "Jz":1.0, "Boundary": periodic, "Seed": seed, "Coupling": coupling_vals, "OTOC_t":values}, ignore_index=True);
    print(count)
    count+=1;

df.to_pickle("/Users/szabo.48/Desktop/OTOC" + "_" + str(L) +".pickle");
print('Completed');
