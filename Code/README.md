## Usage
 
This framework has been tested for the following datasets:
 
### Male (M3) and female (F4) gut microbiomes [1]
To preprocess abundance profile: `preprocess.py`  
To preprocess text embeddings: `sentence_embedding.py`  
Train model using mono-community pipeline: `main_m3_f4_gut_monocomm.py`
 
### Ginger rhizosphere [2]
To preprocess abundance profile: `preprocess_ginger.py`  
To preprocess text embeddings: `sentence_embedding_ginger.py`  
Train model using mono-community pipeline: `main_ginger_monocomm.py`
 
### In vitro gut microbial communities [3]
To preprocess abundance profile and text embeddings: `preprocess_baranwalclark.py`  
Train model using mono-community pipeline: `main_baranwalclark_monocomm.py`  
Train model using multi-community pipeline: `main_baranwalclark_multicomm.py`
 
### Simulated gut microbial communities [4]
To preprocess abundance profile: `preprocess_simulated.py`  
Train model using mono-community pipeline: `main_simulated_monocomm.py`  
Train model using multi-community pipeline: `main_simulated_multicomm.py`
 
### Neonatal gut and respiratory microbial communities [5]
To preprocess abundance profile: `preprocess_grier_all.py`  
To preprocess text embeddings: `sentence_embedding_grier_all.py`  
Train model using mono-community pipeline: `main_grier_monocomm.py`  
Train model using multi-community pipeline: `main_grier_multicomm.py`


## References

[1] Caporaso, J.G., Lauber, C.L., Costello, E.K., Berg-Lyons, D., Gonzalez, A.,
Stombaugh, J., Knights, D., Gajer, P., Ravel, J., Fierer, N., et al.: Moving pictures
of the human microbiome. Genome biology 12(5), 50 (2011)

[2] Wang, C.-W., Michelle Wong, J.-W., Yeh, S.-S., Eric Hsieh, Y., Tseng, C.-H.,
Yang, S.-H., Tang, S.-L.: Soil bacterial community may offer solutions for ginger
cultivation. Microbiology spectrum 10(5), 01803–22 (2022)

[3] Clark, R.L., Connors, B.M., Stevenson, D.M., Hromada, S.E., Hamilton, J.J.,
Amador-Noguez, D., Venturelli, O.S.: Design of synthetic human gut microbiome
assembly and butyrate production. Nature communications 12(1), 3254 (2021)

[4] Baranwal, M., Clark, R.L., Thompson, J., Sun, Z., Hero, A.O., Venturelli, O.S.:
Recurrent neural networks enable design of multifunctional synthetic human gut
microbiome dynamics. Elife 11, 73870 (2022)

[5] Grier, A., McDavid, A., Wang, B., Qiu, X., Java, J., Bandyopadhyay, S., Yang, H.,
Holden-Wiltse, J., Kessler, H.A., Gill, A.L., et al.: Neonatal gut and respiratory
microbiota: coordinated development through time and space. Microbiome 6(1),
193 (2018)


