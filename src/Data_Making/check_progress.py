
def progress_bar(now,n_roop,batch_n=None):

    a="+"*(now+1)
    b=" "*(n_roop-(now+1))
    if batch_n is not None:
        print(f" | No.{batch_n:<3d}",end="")
        
    print(f'|{a}{b}| , ({now+1}/{n_roop}) , {((now+1)/n_roop)*100:>3.0f}% \r',end="")

    

    if n_roop==now+1:
        print()


    