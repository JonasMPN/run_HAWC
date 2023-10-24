import numpy as np
import os
import itertools

global htc_file_name, root_results, case_name, overwrite_existing_case, opt_file_to_be_saved

if __name__ == "__main__":
	hawc_type = "hawc2mb"  # can be 'hawc2s' or 'hawc2mb'
	dir_htc_relates_to = "../data/HAWC/input"  # path from this file to the directory the file paths in the .htc file
	# relate to
	
	htc_file_name = "Redesign_hawc2_flex_2step"  # htc file name w/o extension in the htc dir
	root_results = "../output"
	case_name = "test_hawc2"
	overwrite_existing_case = True
	skip_safety_warning = False  # Danger-setting: if set to True, disable safety check for overwriting cases.
	opt_file_to_be_saved = None
	
	change_param_type = "simultaneously"  # "simultaneously" or "consecutively"
	n_parallel_processes = 4  # check yourself before you shrek yourself. I.e., check your CPU and RAM before you fry
	# your PC with too many processes
	
	# Parameters that ought to be changed must be given as an iterable of the different values they should be. Since, e.g.,
	# "windspeed"'s values are a list, they would need to be specified as a list of lists. If no changes are wanted then
	# every key must be commented out.
	freq = np.round(np.linspace(0.02, 0.05, 2), 2)
	damping = np.round(np.linspace(0.1, 0.9, 3), 1)
	# freq = [0.034, 0.038, 0.042, 0.046]
	# damping = [0.75, 0.8, 0.85, 0.95]
	change_params = {
			"hawcstab2": {
					"operational_data" : {
							# "opt_lambda": np.arange(7, 14).tolist(),
							# "minpitch": np.arange(3, 7).tolist()
					},
					"controller_tuning": {
							# "partial_load": [list(to_list) for to_list in list(itertools.product(freq, damping))],
							# "full_load": [list(to_list) for to_list in list(itertools.product(freq, damping))],
							# "constant_power": [0, 0, 0, 1, 1, 1]
					}
			},
			"dll"      : {
					"type2_dll": {
							"innit": {
									# "constant__7": np.arange(0, 100).tolist()
							}
					}
			}
	}
	# Here, specify as a list of tuples the .opt files that you want to take a slice off (1st entry of each tuple) and as a
	# tuple the wind speed range you want the slice to be (2nd entry of each tuple).
	change_opt_file = [
			# [os.path.join(root_results, case_name, f"opt_lambda_{tsr}/test.opt"), (5, 5)] for tsr in np.arange(7, 14)
	]
	
	case_dir = "../data/HAWC/output/test_hawc2"
	incorporate_ctrl_tuning = [
			os.path.join(root_results, "test_hawc2", dir_name, "test_hawc2_ctrl_tuning.txt")
			for dir_name in os.listdir(case_dir)
	]
	
	hawc2_use_from_hawc2s = [
			("hawcstab2.controller_tuning.constant_power", "dll.type2_dll.init.constant__15"),
	]
# --------------------- end of user input ---------------------


import subprocess
from itertools import zip_longest
import os
import shutil
import pathlib
from lacbox.htc import HTCFile
import lacbox.io as lio
import glob
from multiprocessing import Pool


class IterDict:
	def __init__(self):
		self.lengths = []
		self.consecutively = []
		self.simultaneously = []
		self._blocks = ""
		self._tmp = []
	
	def check_equal_lengths(self, changes):
		self.lengths = []
		
		def iter_for_length(dict_to_check):
			for k, v in dict_to_check.items():
				if isinstance(v, dict):
					iter_for_length(v)
				else:
					try:
						self.lengths.append(len(v))
					except TypeError:
						raise TypeError("The different values need to be collected in a list, but they were given as "
						                f"{type(v)}.")
		
		iter_for_length(changes)
		return all([n_values == self.lengths[0] for n_values in self.lengths]), self.lengths
	
	def to_list(self, changes, change_type):
		implemented = ["consecutively", "simultaneously"]
		if change_type not in implemented:
			raise ValueError(f"'change_type' must be one of {implemented} but was {change_type}.")
		perform = {
				"consecutively" : self._consecutively,
				"simultaneously": self._simultaneously
		}
		return perform[change_type](changes) if any(self.check_equal_lengths(changes)[1]) else [None]
	
	def _consecutively(self, changes):
		self.consecutively = []
		
		def iter_for_values(dict_to_iter):
			for k, v in dict_to_iter.items():
				if isinstance(v, dict):
					self._blocks += f"{k}."
					iter_for_values(v)
				else:
					try:
						for value in v:
							self.consecutively.append([(self._blocks+f"{k}", value)])
					except TypeError:
						raise TypeError("The different values need to be collected in a list, but they were given as "
						                f"{type(v)}.")
			self._blocks = self._blocks[:self._blocks[:-1].rfind(".")+1]
		
		iter_for_values(changes)
		return self.consecutively
	
	def _simultaneously(self, changes):
		self.simultaneously = []
		if len(self.lengths) == 0:
			self.check_equal_lengths(changes)
		n_vars = self.lengths[0]
		
		def iter_for_values(dict_to_iter, index):
			for k, v in dict_to_iter.items():
				if isinstance(v, dict):
					self._blocks += f"{k}."
					iter_for_values(v, index)
				else:
					try:
						iter(v)
						self._tmp.append((self._blocks+f"{k}", v[index]))
					except TypeError:
						raise TypeError("The different values need to be collected in a list, but they were given as "
						                f"{type(v)}.")
			self._blocks = self._blocks[:self._blocks[:-1].rfind(".")+1]
		
		for i in range(n_vars):
			iter_for_values(changes, i)
			self._blocks = ""
			self.simultaneously.append(self._tmp)
			self._tmp = []
		return self.simultaneously


class Utils(IterDict):
	def sort_files(self,
	               results_dir: str,
	               htc_dir: str,
	               htc_file: str,
	               case_name: str,
	               opt_file_to_be_saved: str = None):
		"""
		Grabs all outputs from HAWC and copies them into the results directory. If a file extension of the outputs occurs
		more than once, the respective files are collected in a subdirectory in the results directory. Additionally,
		the used .htc and .opt file can be saved to the results directory, too.
		:param results_dir:
		:param htc_dir:
		:param htc_file:
		:param case_name:
		:param opt_file_to_be_saved:
		:return:
		"""
		shutil.copy(htc_dir+"/"+htc_file+".htc", results_dir+f"/{case_name.replace('/', '_')}.htc")
		if opt_file_to_be_saved is not None:
			shutil.copy(f"opt/{opt_file_to_be_saved}.opt", results_dir+f"/{case_name.replace('/', '_')}.opt")
		
		htc = HTCFile(htc_dir+"/"+htc_file+".htc")
		if "hawcstab2" in htc.keys():
			dirs_with_wanted_data = [htc_dir]
		elif all([block in htc.keys() for block in ["simulation", "output"]]):
			log_file = htc["simulation"]["logfile"].values[0]
			res_file = htc["output"]["filename"].values[0]
			dirs_with_wanted_data = {log_file[:log_file.rfind("/")], res_file[:res_file.rfind("/")]}
		else:
			raise ValueError("Could not figure out whether this is a hawc2s or hawc2mb run. Currently, a hawc2s run is "
			                 "expected to have a 'hawcstab2' block in its .htc file; a hawc2mb run is expected to have a "
			                 "'simulation' and 'output' block.")
		for data_dir in dirs_with_wanted_data:
			files_in_dir = [file for file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, file))]
			file_extensions = {file[file.rfind("."):] for file in files_in_dir if
			                   file[file.rfind("."):] not in [".htc"]}
			for file_extension in file_extensions:
				files_with_current_extension = glob.glob(data_dir+f"/*{file_extension}")
				if len(files_with_current_extension) == 1:
					if "ctrl_tuning" in files_with_current_extension[0] and file_extension != "log":
						case_name += "_ctrl_tuning"
					destination = results_dir+f"/{case_name.replace('/', '_')}{file_extension}"
					shutil.move(files_with_current_extension[0], destination)
				else:
					self.create_dir(results_dir+f"/{file_extension[1:]}")
					for file in files_with_current_extension:
						shutil.move(file, results_dir+f"/{file_extension[1:]}")
			if len(os.listdir(data_dir)) == 0 or data_dir == htc_dir:
				if data_dir != "htc":
					shutil.rmtree(data_dir)
	
	def run(self, dir_htc_relates_to: str, change_params: dict, change_opt_file: list[str],
	        incorporate_ctrl_tuning: list[str], change_param_type: str, n_parallel_processes: int,
	        hawc2_use_from_hawc2s: list[tuple]):
		if not skip_safety_warning:
			if overwrite_existing_case == True:
				if input("'overwrite_existing_case' is set to True. Do you wish to continue? y/n\n") != "y":
					print("Run aborted.")
					exit(0)
		run_dir = os.path.realpath(os.path.dirname(__file__))
		htc_wd = os.path.join(run_dir, dir_htc_relates_to)
		os.chdir(htc_wd)
		if sum([any(self.check_equal_lengths(change_params)[1]),
		        len(change_opt_file) != 0,
		        len(incorporate_ctrl_tuning) != 0]) > 1:
			raise UserWarning(
					"'change_params', 'change_opt_file', and 'incorporate_ctrl_tuning' are not supposed to be "
					"used simultaneously.")
		changes = self.to_list(change_params, change_param_type)
		infos = self.prepare_multiprocessing(n_parallel_processes, htc_wd, hawc2_use_from_hawc2s, changes,
		                                     change_opt_file, incorporate_ctrl_tuning)
		pool = Pool(processes=n_parallel_processes)
		pool.map(self._run_one_processor, infos)
		self.clean(hawc_type)
	
	@staticmethod
	def create_dir(target: str,
	               overwrite: bool = True,
	               add_missing_parent_dirs: bool = True,
	               raise_exception: bool = False,
	               print_message: bool = True) -> tuple[str, bool]:
		"""
		Handles the creation of a directory. Can keep/overwrite directories and creates ones arbitrary meany levels deep.
		:param target: Directory to create
		:param overwrite:
		:param add_missing_parent_dirs: Whether the creation multiple levels deep is allowed.
		:param raise_exception: Whether to stop if an exception occurs.
		:param print_message:
		:return:
		"""
		msg, keep_going = str(), bool()
		try:
			if overwrite:
				if os.path.isdir(target):
					shutil.rmtree(target)
					msg = f"Existing directory {target} was overwritten."
				else:
					msg = f"Could not overwrite {target} as it did not exist. Created it instead."
				keep_going = True
			else:
				msg, keep_going = f"Directory {target} created successfully.", True
			pathlib.Path(target).mkdir(parents=add_missing_parent_dirs, exist_ok=False)
		except Exception as exc:
			if exc.args[0] == 2:  # FileNotFoundError
				if raise_exception:
					raise FileNotFoundError(f"Not all parent directories exist for directory {target}.")
				else:
					msg, keep_going = f"Not all parent directories exist for directory {target}.", False
			elif exc.args[0] == 17:  # FileExistsError
				if raise_exception:
					raise FileExistsError(f"Directory {target} already exists; it was not changed.")
				else:
					msg, keep_going = f"Directory {target} already exists; it was not changed.", False
		if print_message:
			print(msg)
		return msg, keep_going
	
	@staticmethod
	def prepare_multiprocessing(n_processes: int, htc_wd: str, hawc2_use_from_hawc2s: list[tuple], *args: list):
		global htc_file_name, root_results, case_name, overwrite_existing_case, opt_file_to_be_saved
		args_passed = max([len(arg) for arg in args])
		split_size = max(1, args_passed//n_processes)
		n_not_accounted = args_passed-split_size*n_processes
		infos, idx_advanced = [], 0
		for n in range(n_processes if n_not_accounted >= 0 else args_passed):
			process_info = [n]
			idx_start = n*split_size+idx_advanced
			idx_advanced += 1 if n < n_not_accounted else 0
			idx_end = split_size*(n+1)+idx_advanced
			process_info += [arg[idx_start:idx_end] if len(arg) != 0 else [] for arg in args]
			process_info += [htc_wd, hawc_type, htc_file_name, root_results, case_name]
			process_info += [overwrite_existing_case, opt_file_to_be_saved, hawc2_use_from_hawc2s]
			infos.append(process_info)
		return infos
	
	def _run_one_processor(self, info):
		i_run, changes, change_opt_file, incorporate_ctrl_tuning = info[:4]
		htc_wd, hawc_type, htc_file_name, root_results, case_name = info[4:9]
		overwrite_existing_case, opt_file_to_be_saved, hawc2_use_from_hawc2s = info[9:]
		hawc_prep = MoreHAWCIO()
		
		htc_base, case_name_base = htc_file_name, case_name
		for change, opt_file_info, ctrl_tuning_file in zip_longest(changes, change_opt_file, incorporate_ctrl_tuning):
			htc_dir = "htc"
			results_dir = f"{root_results}/{case_name}"
			# if changes wanted, create a temporary .htc directory and file that includes the changes
			if change is not None:
				htc_dir, htc_file_name, name = hawc_prep.tmp_htc(i_run, htc_wd, htc_base, change)
				results_dir += f"/{name}"
			
			if opt_file_info is None and ctrl_tuning_file is None:
				# create (or check for existence) results dir
				self.create_dir(results_dir, overwrite=overwrite_existing_case, raise_exception=True)
			elif opt_file_info is not None:  # prepare the slice of the .opt file and change the .htc accordingly
				name = f"wsp_{opt_file_info[1][0]}_{opt_file_info[1][1]}"
				dir_opt = opt_file_info[0][:opt_file_info[0].rfind("/")]
				results_dir = os.path.join(dir_opt, name)
				self.create_dir(results_dir, overwrite=overwrite_existing_case,
				                raise_exception=True)  # must be before slice_opt()
				htc_dir, htc_file_name = hawc_prep.slice_opt(i_run, htc_wd, results_dir, htc_base, opt_file_info[0],
				                                             opt_file_info[1])
				case_name = case_name_base+f"_wsp_{opt_file_info[1][0]}_{opt_file_info[1][1]}"
			else:  # incorporate control tuning
				ctrl_tuning_file = ctrl_tuning_file.replace("\\", "/")
				ctrl_tuning_dir = ctrl_tuning_file[:ctrl_tuning_file.rfind("/")]
				htc_dir, htc_file_name, case_name = hawc_prep.incorporate_ctrl_tuning(i_run, ctrl_tuning_file, htc_wd,
				                                                                      htc_base)
				results_dir = os.path.join(ctrl_tuning_dir, "simulation_2step")
				self.create_dir(results_dir, overwrite=overwrite_existing_case, raise_exception=True)
				if len(hawc2_use_from_hawc2s) != 0:
					hawc2s_htc_file = glob.glob(os.path.join(ctrl_tuning_dir, "*.htc"))[0]
					hawc_prep.incorporate_hawc2s_into_hawc2(hawc2s_htc_file,
					                                        os.path.join(htc_dir, htc_file_name+".htc"),
					                                        hawc2_use_from_hawc2s)
			
			subprocess.call(f"{hawc_type} {htc_dir}/{htc_file_name}.htc")  # run hawc2s or hacw2mb
			# move and/or copy results and files related to the current HAWC run into the results directory
			self.sort_files(results_dir, htc_dir, htc_file_name, case_name, opt_file_to_be_saved)
	
	@staticmethod
	def clean(hawc_type):
		if hawc_type == "hawc2mb":
			shutil.rmtree(".htc_tmp")


class MoreHAWCIO(Utils):
	def tmp_htc(self, i_run: int, base_dir: str, htc_base_file: str, changes: list[list], name: str = None):
		"""
		Creates a temporary directory with containing an .htc file that contains the wanted changes.
		:param base_dir: Something for HTCFile()
		:param htc_base_file: Name of the original .htc file
		:param change: Which value to change (currently limited to the operational data in the hawcstab2 block. Can
		easily be generalised).
		:param change_opt_file: Change the operational data file that is being read.
		:param name:
		:return:
		"""
		htc = HTCFile(os.path.join("htc", f"{htc_base_file}.htc"), modelpath=base_dir)
		name_from_changes = ""
		for change in changes:
			htc_sub = htc
			blocks, value = change
			for block in blocks.split("."):
				htc_sub = htc_sub[block]
			htc_sub.values = value if type(value) == list else [value]
			name_from_changes += blocks.split(".")[-1]+f"_{self.list_to_str(str(value))}_"
		name = name if name is not None else name_from_changes[:-1]
		
		new_htc_file = f"{htc_base_file}_{name}"
		htc_dir = ".htc_tmp"
		if all([block in htc.keys() for block in ["simulation", "output"]]):  # hawc2 run
			log_file = htc["simulation"]["logfile"].values[0]
			if log_file.rfind("/") == -1:
				raise ValueError("The log file of the HAWC2 simulation must be in an empty directory")
			new_log_file = log_file[:log_file.rfind("/")]+f"_{i_run}"+log_file[log_file.rfind("/"):]
			htc["simulation"]["logfile"].values = [new_log_file]
			
			res_file = htc["output"]["filename"].values[0]
			if res_file.rfind("/") == -1:
				raise ValueError("The output file of the HAWC2 simulation must be in an empty directory")
			new_res_file = res_file[:res_file.rfind("/")]+f"_{i_run}"+res_file[res_file.rfind("/"):]
			htc["output"]["filename"].values = [new_res_file]
			new_htc_file += f"{i_run}"
		else:  # hawc2s run
			htc_dir += f"_{i_run}"
		self.create_dir(htc_dir, overwrite=False)
		
		htc.save(os.path.join(htc_dir, new_htc_file+".htc"))
		return htc_dir, new_htc_file, name
	
	def slice_opt(self, i_run: int, base_dir: str, results_dir: str, htc_base_file: str, opt_file: str,
	              wind_speed_range: tuple):
		"""
		Loads the base .opt file, takes a slice based on the wind speed range, and saves it in the results directory.
		:param base_dir:
		:param results_dir:
		:param htc_base_file:
		:param opt_file:
		:param wind_speed_range:
		:return:
		"""
		name = f"wsp_{wind_speed_range[0]}_{wind_speed_range[1]}"
		
		opt = lio.load_oper(opt_file)
		mask = (opt["ws_ms"] >= wind_speed_range[0]) & (opt["ws_ms"] <= wind_speed_range[1])
		opt = {param: values[mask] for param, values in opt.items()}
		slice_opt_file = os.path.join(results_dir, name+".opt")
		lio.save_oper(slice_opt_file, opt)
		changes = [["hawcstab2.operational_data_filename", slice_opt_file]]
		htc_dir, htc_file, _ = self.tmp_htc(i_run, base_dir, htc_base_file, changes, name)
		return htc_dir, htc_file
	
	def incorporate_ctrl_tuning(self, i_run: int, ctrl_tuning_file: str, base_dir: str, HAWC2S_file: str):
		ctrl_params = self.read_ctrl_tuning(ctrl_tuning_file)
		change_params = {
				"dll": {
						"type2_dll": {
								"init": {
										"constant__11": [[11, float(ctrl_params[1]["K"])]],
										"constant__12": [[12, float(ctrl_params[2]["Kp"])]],
										"constant__13": [[13, float(ctrl_params[2]["Ki"])]],
										"constant__16": [[16, float(ctrl_params[3]["Kp"])]],
										"constant__17": [[17, float(ctrl_params[3]["Ki"])]],
										"constant__21": [[21, float(ctrl_params[3]["K1"])]],
										"constant__22": [[22, float(ctrl_params[3]["K2"])]]
								}}}}
		return self.tmp_htc(i_run, base_dir, HAWC2S_file, self._simultaneously(change_params)[0], "hawc2_ctrl_tuned")
	
	@staticmethod
	def incorporate_hawc2s_into_hawc2(hawc2s_file: str, hawc2_file: str, to_adapt: list[tuple[str]]):
		htc_hawc2s = HTCFile(hawc2s_file)
		htc_hawc2 = HTCFile(hawc2_file)
		for adapt in to_adapt:
			htc2s_param = htc_hawc2s
			for block2s in adapt[0].split("."):
				htc2s_param = htc2s_param[block2s]
			htc2_param = htc_hawc2
			for block2 in adapt[1].split("."):
				htc2_param = htc2_param[block2]
			if "constant__" in block2:
				htc2s_param = [int(block2[block2.rfind("_")+1:]), htc2s_param.values[0]]
			htc2_param.values = htc2s_param
		htc_hawc2.save(hawc2_file)
	
	@staticmethod
	def list_to_str(parameter: str):
		for to_delete in ["[", "]", " "]:
			parameter = parameter.replace(to_delete, "")
		for to_replace in [","]:
			parameter = parameter.replace(to_replace, "_")
		return parameter
	
	@staticmethod
	def read_ctrl_tuning(ctrl_tuning_file: str) -> dict:
		with open(ctrl_tuning_file, "r") as f:
			data = [line.split() for line in f.readlines()]
			read = [  # (line, region, index for parameter name, index for value)
					(1, 1, 0, 2),
					(3, 2, 0, 2),
					(4, 2, 0, 2),
					(5, 2, 0, 2),
					(7, 3, 0, 2),
					(8, 3, 0, 2),
					(9, 3, 0, 2),
					(9, 3, 4, 6),
			]
			extracted = {}
			for line, region, i_param, i_value in read:
				if region not in extracted.keys():
					extracted[region] = {}
				extracted[region][data[line][i_param]] = data[line][i_value]
			return extracted


if __name__ == "__main__":
	Utils().run(dir_htc_relates_to, change_params, change_opt_file, incorporate_ctrl_tuning, change_param_type,
	            n_parallel_processes, hawc2_use_from_hawc2s)

