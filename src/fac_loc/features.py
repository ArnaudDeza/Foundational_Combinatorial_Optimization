import numpy as np





def get_ajwani_UFLP_features(
        transport_cost,                # shape is [num_customer, num_facility]
        facility_LP_solution,          # shape is [num_facility]
        customer_LP_solution           # shape is [num_customer, num_facility]
        ):
    

    num_facilities = len(facility_LP_solution)

    # Feature 1: y_f the LP solution value of the facility variables
    y_f = facility_LP_solution

    # Feature 2: the fractional number of clients covered by f
    # a_f = sum of z_ij for all i in C
    a_f = np.sum(customer_LP_solution,axis=0)

    mu_f_numerators = np.sum(customer_LP_solution * transport_cost, axis=0)  # sum over customers
    mu_f = np.where(
        a_f > 0,
        mu_f_numerators / a_f,
        0.0  # or some other default value, if you prefer
    )

    # Feature 4:  \( \nu_{f,t} = |\{ c \in C : d(c,f) \leq t \cdot \mu_f \}| \) where \( t \in \{2, 4, 8\} \).
    V_f2,V_f4,V_f8 = np.zeros(num_facilities),np.zeros(num_facilities),np.zeros(num_facilities)
    for i in range(num_facilities):
        V_f2[i]=len(np.where(transport_cost[:,i]<=(2*mu_f[i]))[0])
        V_f4[i]=len(np.where(transport_cost[:,i]<=(4*mu_f[i]))[0])
        V_f8[i]=len(np.where(transport_cost[:,i]<=(8*mu_f[i]))[0])

    # Feature 5: \( \beta_{f,t} = |\{ c \in C : d(c,f) \leq t \cdot d_{\text{av}}(c) \}| \) where \( d_{\text{av}}(c) = \frac{\sum_{g \in F} d(c,g) \cdot x_{c,g}}{\alpha_f} \) and \( t \in \{2, 4, 8\} \).
    dav=np.zeros(num_facilities)
    for i in range(num_facilities):
        dav[i]=sum(transport_cost[:,i]*customer_LP_solution[:,i])

    B_f2,B_f4,B_f8=np.zeros(num_facilities),np.zeros(num_facilities),np.zeros(num_facilities)
    for i in range(num_facilities):
        B_f2[i]=len(np.where(transport_cost[:,i]<=(2*dav[i]))[0])
        B_f4[i]=len(np.where(transport_cost[:,i]<=(4*dav[i]))[0])
        B_f8[i]=len(np.where(transport_cost[:,i]<=(8*dav[i]))[0])


    Yf=np.zeros(num_facilities)
    for i in range(num_facilities):
        Yf[i]=(len(np.where(customer_LP_solution[:,i]!=0)[0]))


    # Now create one big matrix with shape [num_facilities, num_features]
    features = np.stack([y_f,a_f,mu_f,V_f2,V_f4,V_f8,B_f2,B_f4,B_f8,Yf],axis=1)
    
    return features