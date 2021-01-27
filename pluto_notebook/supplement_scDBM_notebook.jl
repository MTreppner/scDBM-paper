### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# ╔═╡ e53e9d5a-6091-11eb-3582-398334c98d77
begin
	# Load required packages
	using Pkg, HTTP, CSV, DataFrames, Random, Distances, 
        UMAP, Clustering, Gadfly, DelimitedFiles, 
        MultivariateStats, Cairo, Fontconfig, StatsBase

	# Set default plot size
    set_default_plot_size(50cm, 20cm)
	
	# Set working directory
	cd("/Users/martintreppner/Desktop")
end;

# ╔═╡ f1a368be-6091-11eb-0da3-77b12261da8c
begin
	# Install and Load single-cell deep Boltzmann machine (scDBM) 
	# Pkg.add(PackageSpec(url="https://github.com/MTreppner/scDBM.jl", rev="master"))
	using scDBM
end;

# ╔═╡ c5044126-60b8-11eb-1d18-33c34995a8d5
begin
	using PyCall
	
	n_genes = 2000
	# Load python packages
	random = pyimport("random")
	os = pyimport("os")
	np = pyimport("numpy")
	pd = pyimport("pandas")
	scvi = pyimport("scvi")
	scvi_dataset = pyimport("scvi.dataset")
	scvi_models = pyimport("scvi.models")
	scvi_inference = pyimport("scvi.inference")

	dir = "notebook_test/segerstolpe_hvg.csv";
    
	countmatrix_scvi = scvi_dataset.CsvDataset(dir, save_path = "", new_n_genes = n_genes);    
	# Set hyperparameters
	n_latent=14
	n_hidden=64
	n_layers=2
	n_epochs=200
	lr_scvi=0.01
	use_batches = false
	use_cuda = true

	# Train the model and output model likelihood every epoch
	vae = scvi_models.VAE(countmatrix_scvi.nb_genes, 
		n_batch=countmatrix_scvi.n_batches * use_batches, 
		n_latent=n_latent, 
		n_hidden=n_hidden,
		n_layers=n_layers, 
		reconstruction_loss="nb", 
		dispersion = "gene"
	);

	trainer = scvi_inference.UnsupervisedTrainer(vae,
		countmatrix_scvi,
		train_size=0.7,
		use_cuda=use_cuda,
		frequency=1,
		n_epochs_kl_warmup=75,
		batch_size=36
	);

	trainer.train(n_epochs=n_epochs,lr=lr_scvi)

	full = trainer.create_posterior(trainer.model, countmatrix_scvi);

	batch_size = 32
	n_cells = 16

	# Generate Data from posterior
	gen_data_scvi = full.generate(batch_size=batch_size,n_samples=10, sample_prior=false, n_cells=n_cells);
	gen_data_scvi_orig = Array{Float64,2}(gen_data_scvi[1][:,:,2]);

####### scVI Zero data 

	n_genes = 2000
	# Load python packages
	random = pyimport("random")
	os = pyimport("os")
	np = pyimport("numpy")
	pd = pyimport("pandas")
	scvi = pyimport("scvi")
	scvi_dataset = pyimport("scvi.dataset")
	scvi_models = pyimport("scvi.models")
	scvi_inference = pyimport("scvi.inference")

	dir = "notebook_test/segerstolpe_hvg_zeros.csv";
    
	countmatrix_scvi = scvi_dataset.CsvDataset(dir, save_path = "", new_n_genes = n_genes);    

	# Set hyperparameters
	n_latent=14
	n_hidden=64
	n_layers=2
	n_epochs=200
	lr_scvi=0.01
	use_batches = false
	use_cuda = true

	# Train the model and output model likelihood every epoch
	vae = scvi_models.VAE(countmatrix_scvi.nb_genes, 
		n_batch=countmatrix_scvi.n_batches * use_batches, 
		n_latent=n_latent, 
		n_hidden=n_hidden,
		n_layers=n_layers, 
		reconstruction_loss="nb", 
		dispersion = "gene"
	);

	trainer = scvi_inference.UnsupervisedTrainer(vae,
		countmatrix_scvi,
		train_size=0.7,
		use_cuda=use_cuda,
		frequency=1,
		n_epochs_kl_warmup=75,
		batch_size=36
	);

	trainer.train(n_epochs=n_epochs,lr=lr_scvi)

	full = trainer.create_posterior(trainer.model, countmatrix_scvi);

	batch_size = 32
	n_cells = 16

	# Generate Data from posterior
	gen_data_scvi = full.generate(batch_size=batch_size,n_samples=10, sample_prior=false, n_cells=n_cells);
	gen_data_scvi_zeros = Array{Float64,2}(gen_data_scvi[1][:,:,2]);

end;

# ╔═╡ a6314f18-60a5-11eb-0307-450764cbf160
md"
#### Supplementary Material:  

##### Synthetic Single-Cell RNA-Sequencing Data from Small Pilot Studies using Deep Generative Models
"

# ╔═╡ 81a8c5a8-6092-11eb-3414-69625042ef17
md"
###### scDBM on original data
"

# ╔═╡ f6e0abde-6091-11eb-2a1c-010ab8a51452
begin
	global countmatrix = CSV.read("notebook_test/segerstolpe_hvg.csv")
	global countmatrix = Array{Float64,2}(Array{Float64,2}(countmatrix[:,2:end])')

	# Sample pilot dataset
	pilot_data = countmatrix;

	# Train- Test-split
	Random.seed!(101);
	data, datatest = splitdata(pilot_data, 0.3);
	datadict = DataDict("Training data" => data, "Test data" => datatest);

	# Set hyperparameters
	epochs = 60;                                    
	init_disp = (ones(size(data,2)) .* 0.5);     
	lr = [1.0e-11*ones(6000);0.00005*ones(1000)];   
	regularization = 2.5;                          

	# Train scDBM
	Random.seed!(59);
	global dbm = fitdbm(data, 
		epochs = 10, 
		learningrate = 1.0e-11, 
		batchsizepretraining = 16,
		  pretraining = [
				# Negative-Binomial RBM
				TrainLayer(nhidden = 64,
				learningrates = lr, 
				epochs = epochs,
				rbmtype = NegativeBinomialBernoulliRBM, 
				inversedispersion = init_disp,
				fisherscoring = 1,
				lambda = regularization),
				# Bernoulli RBM
				TrainLayer(nhidden = 16, 
				learningrate = 1.0e-4,
				epochs = 40)]
		)	
	
	# Generate synthetic cells
	Random.seed!(559);
	number_gensamples = 2090#size(pilot_data,1);
	synthetic_cells = initparticles(dbm, number_gensamples);
	gibbssamplenegativebinomial!(pilot_data,
		synthetic_cells, 
		dbm, 
		30
	);
end;

# ╔═╡ a02127d2-6092-11eb-26fc-e749e38caaca
begin
	using Distributions
	global countmatrix_impute = CSV.read("notebook_test/segerstolpe_hvg.csv")
	global countmatrix_impute = Array{Float64,2}(Array{Float64,2}(countmatrix_impute[:,2:end])')

	
	Random.seed!(112);
	tmp = rand.(Uniform(0,1), size(countmatrix_impute,1), size(countmatrix_impute,2))
	tmp1 = tmp .> 0.2
	tmp_zero_proportion = mapslices(sum, tmp1, dims=1) ./ size(countmatrix_impute,1)

	countmatrix_impute = countmatrix_impute .* tmp1

	tmp = DataFrame(countmatrix_impute')
	CSV.write("notebook_test/segerstolpe_hvg_zeros.csv", tmp, writeheader = true)

	# Sample pilot dataset
	pilot_data_zeros = countmatrix_impute;

	# Train- Test-split
	Random.seed!(101);
	data_zeros, datatest_zeros = splitdata(pilot_data_zeros, 0.3);
	datadict_zeros = DataDict("Training data" => data_zeros, 
		"Test data" => datatest_zeros);

	# Set hyperparameters
	epochs_zeros = 60;                                    
	init_disp_zeros = (ones(size(data_zeros,2)) .* 0.5);     
	lr_zeros = [1.0e-11*ones(6000);0.00005*ones(1000)];   
	regularization_zeros = 2.5;                          

	# Train scDBM
	Random.seed!(59);
	global dbm_zeros = fitdbm(data_zeros, 
		epochs = 10, 
		learningrate = 1.0e-11, 
		batchsizepretraining = 16,
		  pretraining = [
				# Negative-Binomial RBM
				TrainLayer(nhidden = 64,
				learningrates = lr_zeros, 
				epochs = epochs_zeros,
				rbmtype = NegativeBinomialBernoulliRBM, 
				inversedispersion = init_disp_zeros,
				fisherscoring = 1,
				lambda = regularization_zeros),
				# Bernoulli RBM
				TrainLayer(nhidden = 16, 
				learningrate = 1.0e-4,
				epochs = 40)]
		)	
	
	# Generate synthetic cells
	Random.seed!(559);
	number_gensamples_zeros = 2090
	synthetic_cells_zeros = initparticles(dbm_zeros, number_gensamples);
	gibbssamplenegativebinomial!(pilot_data_zeros,
		synthetic_cells_zeros, 
		dbm_zeros, 
		30
	);
end;

# ╔═╡ 9509ec9e-6092-11eb-2157-a540547604cb
md"
###### scDBM on original data with added zeros
"

# ╔═╡ e40e07ce-6094-11eb-1137-b5acb0fec21a
md"
###### scVI
"

# ╔═╡ e5f44554-60a2-11eb-33ab-e199bec0c726
begin
	cd("/Users/martintreppner/Desktop")
	
	# scDBM
	scdbm_out = DataFrame(synthetic_cells[1])
	CSV.write("notebook_test/imputation_scdbm.csv", scdbm_out, writeheader = true)

	scdbm_zeros_out = DataFrame(synthetic_cells_zeros[1])
	CSV.write("notebook_test/imputation_scdbm_zeros.csv", 
		scdbm_zeros_out, 
		writeheader = true)

	# scVI
	scvi_out = DataFrame(gen_data_scvi_orig)
	CSV.write("notebook_test/imputation_scvi.csv", 
		scvi_out,
		writeheader = true)

	scvi_zeros_out = DataFrame(gen_data_scvi_zeros)
	CSV.write("notebook_test/imputation_scvi_zeros.csv", 
		scvi_zeros_out, 
		writeheader = true)
end;

# ╔═╡ Cell order:
# ╟─a6314f18-60a5-11eb-0307-450764cbf160
# ╠═e53e9d5a-6091-11eb-3582-398334c98d77
# ╠═f1a368be-6091-11eb-0da3-77b12261da8c
# ╟─81a8c5a8-6092-11eb-3414-69625042ef17
# ╠═f6e0abde-6091-11eb-2a1c-010ab8a51452
# ╟─9509ec9e-6092-11eb-2157-a540547604cb
# ╠═a02127d2-6092-11eb-26fc-e749e38caaca
# ╟─e40e07ce-6094-11eb-1137-b5acb0fec21a
# ╠═c5044126-60b8-11eb-1d18-33c34995a8d5
# ╠═e5f44554-60a2-11eb-33ab-e199bec0c726
