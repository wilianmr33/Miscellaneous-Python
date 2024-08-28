import numpy as np
import pandas as pd
import panther as pt

def dist_bad(base, fxs, resp):
    df_ = base.groupby([fxs])[resp].agg({'sum','size'}).reset_index().rename({'sum': 'Maus', 'size': 'Total'}, axis=1)
    df_['Bons'] = df_['Total'] - df_['Maus']
    df_['% Bad'] = df_['Maus']/df_['Total']
    df_['% Total'] = df_['Total']/sum(df_['Total'])

    df_ = df_[[fxs,'Bons','Maus','Total', '% Bad', '% Total']]
    return df_

def fx_scr(base, score):
    base['FX_'+score] =   np.where( base[score].isna(),  'A. Missing',
                          np.where( base[score]  <    0, 'A. SCR <   0',
                          np.where( base[score]  <= 100, 'A. 000 A 100',
                          np.where( base[score]  <= 200, 'B. 101 A 200',
                          np.where( base[score]  <= 300, 'C. 201 A 300',
                          np.where( base[score]  <= 400, 'D. 301 A 400',
                          np.where( base[score]  <= 500, 'E. 401 A 500',
                          np.where( base[score]  <= 600, 'F. 501 A 600',
                          np.where( base[score]  <= 700, 'G. 601 A 700',
                          np.where( base[score]  <= 800, 'H. 701 A 800',
                          np.where( base[score]  <= 900, 'I. 801 A 900', 
                          np.where( base[score] <= 1000, 'J. 901 A 1000', 'H. SCORE ESTRANHO, VERIFIQUE'))))))))))))    
    
# Função para comparação de KS de multiplos scores - Com grupos
def comparaKS( df, target_column, scores_columns, group_column ):
    
    # Cálculo do KS por safra
    if isinstance(scores_columns, list) & (len(scores_columns) == 1):
        return pt.stats.stats.ks_metric( df, target_column, scores_columns[0], group_column)
    elif isinstance(scores_columns, str):
        return pt.stats.stats.ks_metric( df, target_column, scores_columns, group_column)
    else:
        ks = pd.DataFrame()
        for i, score in enumerate(scores_columns):
            ks_ = pt.stats.stats.ks_metric( df, target_column, score, group_column)\
                .rename(columns = {'KS': score})
            if i == 0:
                ks = ks_.copy()
            else:
                ks = ks.merge(ks_, how = 'left', on = group_column)
            
        return ks

# Função para comparação de KS de multiplos scores - Sem Grupos
def stck_ks(base, scr_vector):
    ks_values = []
    for i in scr_vector:
        df_ = pt.stats.stats.ks_metric( base, 'CONCEITO', i )
        ks_values.append(df_)
    
    DF2 = pd.DataFrame([ks_values], columns = scr_vector)

    return DF2

# função dist_bad + media de duas variáveis desejadas
def dist_bad3(base, fxs, resp, ohts_):
    
    """ 
    Colunas da lista oths_ estão desconsiderando os 
    valores especiais na hora de computar a 
    métrica solicidada (mediana/soma)
     # Ex.: dist_bad3(base = df_categs2[ (df_categs2['DT_T0'] == '2022-12-01')], fxs = 'risk_groups', resp ='CONCEITO', ohts_ =['TVLFATURA30DPG', 'TVLFATURA30D', 'TPONTQTEMPFIN'])
    """
    base2 = base.copy()
    base2[ohts_] = base2[ohts_].replace([-1, -2], np.nan)

    base2['flag_valor_espec'] = np.where(base2['TVLFATURA30D'].isin([-1, -2]), 1, 0)

    d = {**{resp: ['size', 'sum']},
         **{'flag_valor_espec': ['mean']},
         **dict.fromkeys(ohts_, ['sum', 'median'])
    }
        
    df_ = base2.groupby([fxs]).agg(d).reset_index()
    df_.columns = df_.columns.map('_'.join)
    df_ = df_.rename({resp+'_sum': 'Maus', resp+'_size': 'Total'}, axis=1)
    df_['Bons'] = df_['Total'] - df_['Maus']
    df_['% Bad'] = df_['Maus']/df_['Total']
    df_['% Total'] = df_['Total']/sum(df_['Total'])
    df_['%_Pgto'] = df_['TVLFATURA30DPG_sum']/df_['TVLFATURA30D_sum']
    
    oth_vars = list(df_.filter(regex='_median$',axis=1).columns) # Pegando variáveis terminadas em median
    df_ = df_[[fxs+'_', 'Bons', 'Maus', 'Total', '% Bad', '% Total'] + ['%_Pgto'] + oth_vars + ['flag_valor_espec_mean']]
    return df_
