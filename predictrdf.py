import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import pyPRISM

def cal_rdf (x0, x1, x2, x3, x4):
    def interpolate_guess(domain_from,domain_to,rank,guess):
    # Helper for upscaling the intial guesses
        guess = guess.reshape((domain_from.length,rank,rank))
        new_guess = np.zeros((domain_to.length,rank,rank))
        for i in range(rank):
            for j in range(rank):
                new_guess[:,i,j] = np.interp(domain_to.r,domain_from.r,guess[:,i,j])
        return new_guess.reshape((-1,))

    d = 1.0 #polymer segment diameter (đường kính của hạt polymer)
    phi = x3 #volume fraction of nanoparticles (number density)
    eta = 0.4 #total occupied volume fraction
    
    sys = pyPRISM.System(['particle','polymer'],kT=1.0)
    sys.domain = pyPRISM.Domain(dr=0.1,length=1024)
    
    guess = np.zeros(sys.rank*sys.rank*sys.domain.length)
    N = x4
    #Interaction Between Polymer and NP(nano)
    epsilon1 = x0
    #Interaction Between NP and NP
    epsilon2 = x1
    #đường kính hạt nano
    D = x2
    
    sys.diameter['polymer'] = d
    sys.diameter['particle'] = D
    sys.density['polymer'] = (1-phi)*eta/sys.diameter.volume['polymer']
    sys.density['particle'] = phi*eta/sys.diameter.volume['particle']
    print('--> rho=',sys.density['polymer'],sys.density['particle'])
    
    sys.omega['polymer','polymer'] = pyPRISM.omega.FreelyJointedChain(length=N,l=4.0*d/3.0)
    sys.omega['polymer','particle'] = pyPRISM.omega.InterMolecular()
    sys.omega['particle','particle'] = pyPRISM.omega.SingleSite()
    
    
    sys.potential['polymer','polymer'] = pyPRISM.potential.HardSphere()
    sys.potential['polymer','particle'] = pyPRISM.potential.Exponential(alpha=0.5,epsilon=epsilon1)
    sys.potential['particle','particle'] = pyPRISM.potential.Exponential(alpha=0.5,epsilon=epsilon2)
    
    sys.closure['polymer','polymer'] = pyPRISM.closure.PercusYevick()
    sys.closure['polymer','particle'] = pyPRISM.closure.PercusYevick()
    sys.closure['particle','particle'] = pyPRISM.closure.HyperNettedChain()

    PRISM = sys.createPRISM()
    result = PRISM.solve(guess)
    guess = np.copy(PRISM.x)
    last_guess=guess
    
    sys = pyPRISM.System(['particle','polymer'],kT=1.0)
    sys.domain = pyPRISM.Domain(dr=0.075,length=2048)
    
    sys.diameter['polymer'] = d
    sys.diameter['particle'] = D
    sys.density['polymer'] = (1-phi)*eta/sys.diameter.volume['polymer']
    sys.density['particle'] = phi*eta/sys.diameter.volume['particle']
    
    sys.omega['polymer','polymer'] = pyPRISM.omega.FreelyJointedChain(length=N,l=4.0*d/3.0)
    sys.omega['polymer','particle'] = pyPRISM.omega.NoIntra()
    sys.omega['particle','particle'] = pyPRISM.omega.SingleSite()
    
    sys.closure['polymer','polymer'] = pyPRISM.closure.PercusYevick()
    sys.closure['polymer','particle'] = pyPRISM.closure.PercusYevick()
    sys.closure['particle','particle'] = pyPRISM.closure.HyperNettedChain()
    
    gr_results = []
    guess = interpolate_guess(pyPRISM.Domain(dr=0.1,length=1024),sys.domain,sys.rank,last_guess)
    sys.potential['polymer','polymer'] = pyPRISM.potential.HardSphere()
    sys.potential['polymer','particle'] = pyPRISM.potential.Exponential(alpha=0.5,epsilon=epsilon1)
    sys.potential['particle','particle'] = pyPRISM.potential.Exponential(alpha=0.5,epsilon=epsilon2)
    
    PRISM = sys.createPRISM()
    result = PRISM.solve(guess)
    
    x = sys.domain.r
    y = pyPRISM.calculate.pair_correlation(PRISM)['particle','particle']
    for j in range (len(x)):
        gr_results.append([x[j],y[j]])
    data_df = pd.DataFrame(gr_results)
    return data_df

st.title('nanoNET: machine learning platform for predicting nanoparticles distribution in a polymer matrix')

st.header('Problem')
col1, col2 = st.columns([3,3])
with col1: 
    st.image('image1.png')
with col2: 
    st.image('image2.png')

st.header('Prediction')
st.subheader('Select parameters')
col0, col1, col2, col3, col4 = st.columns(5)
with col0:
    X_0 = st.number_input('NP-polymers', 0.1, 1.5)
    X_1 = st.number_input('NP-NP', 0.1, 1.5)
    X_2 = st.number_input('D', 2, 5)
    X_3 = st.number_input('$Phi*10^{-3}$', 1, 5)
    X_4 = st.number_input('N', 25, 40)
    
btn = st.button('Calculater')
if btn:
    actual_rdf = cal_rdf(X_0, X_1, X_2, X_3/(1000), X_4)
    X_5 = actual_rdf.T.iloc[0]
    #load model
    loaded_rf = joblib.load("my_rf.pkl")
        
    X = []
    for j in range (len(X_5)):
        X.append([X_0, X_1, X_2, X_3/(1000), X_4])

    X = pd.DataFrame(X)
    X['r/sigma'] = X_5

    X = np.array(X)
    rdf_predict = loaded_rf.predict(X)
    
    #rdf = pd.DataFrame({'$r/\sigma$': X_5.T.iloc[0], 'Predicted RDF': rdf_predict})
    rdf = pd.DataFrame({'r/sigma': actual_rdf.T.iloc[0], 'Actual RDF' : actual_rdf.T.iloc[1], 'Predicted RDF': rdf_predict})

    st.subheader('Predict RDF of polymer')
    st.dataframe(rdf)

    @st.cache_data 
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(rdf)
    st.download_button(label = 'Download predict rdf', data = csv, file_name = 'rdf.csv', mime = 'text/csv', key = 'download-csv')

    st.header('Plot RDF')
    fig, ax = plt.subplots(figsize=(4,4))
    plt.plot(rdf.T.iloc[0], rdf.T.iloc[1], 'cyan', linestyle='-', linewidth=2, marker='', label=f'Actual RDF')
    plt.plot(rdf.T.iloc[0], rdf.T.iloc[2],'r', linestyle='--', linewidth=2, marker='', label=f'Predicted RDF')

    ax = plt.gca()
    ax.set_xlim([1,15])
    ax.set_ylim([-1,10])
    plt.tick_params(direction = 'in')

    plt.xlabel(r'$r/\sigma$')
    plt.ylabel('g(r)')

    #chú thích
    legend = plt.legend(title='', loc='best')
    legend.get_title().set_multialignment('center')
    plt.savefig(f"rdf.png")
    st.plotly_chart(fig, use_container_width=True)
    with open('rdf.png', 'rb') as file:
        st.download_button(label = 'Download the graph', data = file, file_name = 'rdf.png', mime = 'image/png')
