from utils import *

def maxz(table):
    while next_round_r(table)==True:
        table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)
    while next_round(table)==True:
        table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)        
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    i = 0
    val = {}
    for i in range(var):
        col = table[:,i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]            
            val[gen_var(table)[i]] = table[loc,-1]
        else:
            val[gen_var(table)[i]] = 0
    val['max'] = table[-1,-1]
    return val
    
def minz(table):
    table = convert_min(table)
    while next_round_r(table)==True:
        table = pivot(loc_piv_r(table)[0],loc_piv_r(table)[1],table)    
    while next_round(table)==True:
        table = pivot(loc_piv(table)[0],loc_piv(table)[1],table)       
    lc = len(table[0,:])
    lr = len(table[:,0])
    var = lc - lr -1
    i = 0
    val = {}
    for i in range(var):
        col = table[:,i]
        s = sum(col)
        m = max(col)
        if float(s) == float(m):
            loc = np.where(col == m)[0][0]             
            val[gen_var(table)[i]] = table[loc,-1]
        else:
            val[gen_var(table)[i]] = 0 
            val['min'] = table[-1,-1]*-1
    return val
    
if __name__ == "__main__":
    m = gen_matrix(2,2)
    constrain(m,'2,-1,G,10')
    constrain(m,'1,1,L,20')
    obj(m,'5,10,0')
    print(maxz(m))     
    m = gen_matrix(2,4)
    constrain(m,'2,5,G,30')
    constrain(m,'-3,5,G,5')
    constrain(m,'8,3,L,85')
    constrain(m,'-9,7,L,42')
    obj(m,'2,7,0')
    print(minz(m))

    m = gen_matrix(3,2)
    constrain(m,'-1,-1,-1,L,-2')
    constrain(m,'2,-1,1,L,1')
    obj(m,'2,-6,0,0')
    print(maxz(m))  
