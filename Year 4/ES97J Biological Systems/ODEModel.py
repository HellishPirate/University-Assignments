import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class ES97J_2110409_A1:

    def __init__(self): # 1
        
        self.alpha = 2
        self.eta = 1
        self.k = 1
        
        self.K = 0 # proportional gain
        self.eta2 = 0 # param for integer control
        self.theta = 0 # integral gain
        self.dilution = 0

        self.beta0 = 0.5
        self.beta1 = 1
        self.beta2 = 2
        self.beta3 = 1
        self.beta4 = 2
        self.beta5 = 1
        self.beta6 = 3
        self.beta7 = 1
        
        self.delta1 = 0
        self.delta2 = 0

        self.muVec = np.array([5, 10, 2, 0.5, 7])
        self.muTimePoints = np.arange(0, 240 + 1, 48)
        self.muVecPointer = 0
        self.timePoints = np.arange(0, 138 + 1, 6)

        self.X0 = np.ones((6,)) * 1E-09

        self.assignmentModelOptions = {
            "mu": self.muVec[self.muVecPointer],
            # "Delta1": 0,
            # "Delta2": 0,
            # "Control": "None"
        }

        self.x1Exp = np.array([2.88,2.73,2.70,2.81,2.84,2.75,2.68,2.60,3.90,3.78,3.69,3.98,
                               3.77,3.85,3.95,3.82,1.84,1.69,1.81,1.82,1.71,1.54,1.66])
        self.x5Exp = np.array([2.72,2.66,2.62,2.74,2.76,2.67,2.61,2.53,5.11,5.07,4.98,5.27,
                               5.06,5.14,5.24,5.11,1.41,1.13,1.24,1.25,1.14,0.97,1.10])
        
        self.figCounter = 1

    def assignmentModel(self, T, X): # 2

        z1, x1, x2, x3, x4, x5 = X
        mu = self.assignmentModelOptions["mu"]

        x1dot = 2*self.beta2*x2 - 2*self.beta1*x1**2 + self.alpha*(self.k*z1-self.K*x5)
        x2dot = (self.beta1+self.delta1)*x1**2 - self.beta2*x2 - self.beta3*x2
        x3dot = self.beta0 - self.beta4*x3 - self.beta5*x3
        x4dot = (self.beta3+self.delta2)*x2 + self.beta4*x3 - self.beta6*x4
        x5dot = self.beta5*x3 + self.beta6*x4 - self.beta7*x5
        z1dot = mu - self.eta*z1

        return np.array([z1dot, x1dot, x2dot, x3dot, x4dot, x5dot])
    
    def assignmentModelX4(self, T, X): # 2

        z1, x1, x2, x3, x4, x5 = X
        mu = self.assignmentModelOptions["mu"]

        x1dot = 2*self.beta2*x2 - 2*self.beta1*x1**2 + self.alpha*(self.k*z1-self.K*x4)
        x2dot = (self.beta1+self.delta1)*x1**2 - self.beta2*x2 - self.beta3*x2
        x3dot = self.beta0 - self.beta4*x3 - self.beta5*x3
        x4dot = (self.beta3+self.delta2)*x2 + self.beta4*x3 - self.beta6*x4
        x5dot = self.beta5*x3 + self.beta6*x4 - self.beta7*x5
        z1dot = mu - self.eta*z1

        return np.array([z1dot, x1dot, x2dot, x3dot, x4dot, x5dot])
    
    def assignmentModelAntithetic(self, T, X): # 3
        
        z1, x1, x2, x3, x4, x5, z2 = X
        mu = self.assignmentModelOptions["mu"]

        x1dot = 2*self.beta2*x2 - 2*self.beta1*x1**2 + self.alpha*(self.k*z1-self.K*x5) - self.dilution*x1
        x2dot = (self.beta1+self.delta1)*x1**2 - self.beta2*x2 - self.beta3*x2 - self.dilution*x2
        x3dot = self.beta0 - self.beta4*x3 - self.beta5*x3 - self.dilution*x3
        x4dot = (self.beta3+self.delta2)*x2 + self.beta4*x3 - self.beta6*x4 - self.dilution*x4
        x5dot = self.beta5*x3 + self.beta6*x4 - self.beta7*x5 - self.dilution*x5
        z1dot = mu - self.eta2*z1*z2 - self.dilution*z1
        z2dot = self.theta*x5 - self.eta2*z1*z2 - self.dilution*z2

        return np.array([z1dot, x1dot, x2dot, x3dot, x4dot, x5dot, z2dot])
    
    def assignmentModelAntitheticX4(self, T, X): # 4
        
        z1, x1, x2, x3, x4, x5, z2 = X
        mu = self.assignmentModelOptions["mu"]

        x1dot = 2*self.beta2*x2 - 2*self.beta1*x1**2 + self.alpha*(self.k*z1-self.K*x4)
        x2dot = (self.beta1+self.delta1)*x1**2 - self.beta2*x2 - self.beta3*x2
        x3dot = self.beta0 - self.beta4*x3 - self.beta5*x3
        x4dot = (self.beta3+self.delta2)*x2 + self.beta4*x3 - self.beta6*x4
        x5dot = self.beta5*x3 + self.beta6*x4 - self.beta7*x5
        z1dot = mu - self.eta2*z1*z2
        z2dot = self.theta*x4 - self.eta2*z1*z2

        return np.array([z1dot, x1dot, x2dot, x3dot, x4dot, x5dot, z2dot])

    def assignmentModelCost(self, beta3): # 5
        
        self.beta3 = beta3

        x1Final = np.zeros(len(self.timePoints) - 1)
        x5Final = np.zeros(len(self.timePoints) - 1)

        self.X0 = np.ones((6,)) * 1E-09
        self.muVecPointer = 0
        self.assignmentModelOptions["mu"] = self.muVec[self.muVecPointer]

        for i in range(len(self.timePoints)-1):
            
            sol = integrate.solve_ivp(self.assignmentModel, 
                                      [self.timePoints[i], self.timePoints[i + 1]], 
                                      self.X0)
            X = sol.y

            x1Final[i] = X[1, -1]
            x5Final[i] = X[5, -1]

            if self.timePoints[i + 1] in self.muTimePoints:
                self.muVecPointer += 1
                if self.muVecPointer < len(self.muVec):
                    self.assignmentModelOptions["mu"] = self.muVec[self.muVecPointer]
                    
            self.X0 = X[:, -1]

        rmse = (np.sqrt(np.mean((self.x1Exp - x1Final)**2)) + 
                np.sqrt(np.mean((self.x5Exp - x5Final)**2)))
        return rmse
    
    def runAssignmentModel(self, delta1=0, delta2=0, delta2_time=0): # 6
        
        self.X0 = np.ones((6, )) * 1E-09
        
        x1 = np.array([]); x5 = np.array([])
        time = np.array([])
        fullMuSig = np.array([])
        
        for i in range(len(self.muTimePoints)-1):
            
            self.assignmentModelOptions["mu"] = self.muVec[i]
            
            if self.muTimePoints[i] < 144:
                self.delta1 = 0
            elif self.muTimePoints[i] >= 144:
                self.delta1 = delta1
            
            if self.muTimePoints[i] < delta2_time:
                self.delta2 = 0
            elif self.muTimePoints[i] >= delta2_time:
                self.delta2 = delta2
                
            sol = integrate.solve_ivp(self.assignmentModel, 
                                      [self.muTimePoints[i], self.muTimePoints[i+1]], 
                                      self.X0,)
            T = sol.t; X = sol.y
            
            x1 = np.append(x1, X[1,:]); x5 = np.append(x5, X[5,:])
            time = np.append(time, T)
            fullMuSig = np.append(fullMuSig, np.repeat(self.muVec[i], len(T)))
            
            self.X0 = X[:,-1]
            
        return x1, x5, time, fullMuSig
    
    def runAssignmentModelX4(self, delta1=0, delta2=0, delta2_time=0): # 6
        
        self.X0 = np.ones((6, )) * 1E-09
        
        fullX = np.empty((6,0))
        time = np.array([])
        fullMuSig = np.array([])
        
        for i in range(len(self.muTimePoints)-1):
            
            self.assignmentModelOptions["mu"] = self.muVec[i]
            
            if self.muTimePoints[i] < 144:
                self.delta1 = 0
            elif self.muTimePoints[i] >= 144:
                self.delta1 = delta1
            
            if self.muTimePoints[i] < delta2_time:
                self.delta2 = 0
            elif self.muTimePoints[i] >= delta2_time:
                self.delta2 = delta2
                
            sol = integrate.solve_ivp(self.assignmentModelX4, 
                                      [self.muTimePoints[i], self.muTimePoints[i+1]], 
                                      self.X0,)
            T = sol.t; X = sol.y
            
            fullX = np.hstack((fullX, X))
            time = np.append(time, T)
            fullMuSig = np.append(fullMuSig, np.repeat(self.muVec[i], len(T)))
            
            self.X0 = X[:,-1]
            
        return fullX.T, time, fullMuSig
    
    def runAssignmentModelAntithetic(self, delta1=0, delta2=0, delta2_time=0): # 7
        
        self.X0 = np.ones((7, )) * 1E-09
        
        x1 = np.array([]); x5 = np.array([])
        time = np.array([])
        fullMuSig = np.array([])
        
        for i in range(len(self.muTimePoints)-1):
            
            self.assignmentModelOptions["mu"] = self.muVec[i]
            
            if self.muTimePoints[i] < 144:
                self.delta1 = 0
            elif self.muTimePoints[i] >= 144:
                self.delta1 = delta1
            
            if self.muTimePoints[i] < delta2_time:
                self.delta2 = 0
            elif self.muTimePoints[i] >= delta2_time:
                self.delta2 = delta2
                
            sol = integrate.solve_ivp(self.assignmentModelAntithetic, 
                                      [self.muTimePoints[i], self.muTimePoints[i+1]], 
                                      self.X0,)
            T = sol.t; X = sol.y
            
            x1 = np.append(x1, X[1,:]); x5 = np.append(x5, X[5,:])
            time = np.append(time, T)
            fullMuSig = np.append(fullMuSig, np.repeat(self.muVec[i], len(T)))
            
            self.X0 = X[:,-1]
            
        return x1, x5, time, fullMuSig
    
    def runAssignmentModelAntitheticX4(self, delta1=0, delta2=0, delta2_time=0): # 8
        
        self.X0 = np.ones((7, )) * 1E-09
        
        fullX = np.empty((7,0))
        time = np.array([])
        fullMuSig = np.array([])
        
        for i in range(len(self.muTimePoints)-1):
            
            self.assignmentModelOptions["mu"] = self.muVec[i]
            
            if self.muTimePoints[i] < 144:
                self.delta1 = 0
            elif self.muTimePoints[i] >= 144:
                self.delta1 = delta1
            
            if self.muTimePoints[i] < delta2_time:
                self.delta2 = 0
            elif self.muTimePoints[i] >= delta2_time:
                self.delta2 = delta2
                
            sol = integrate.solve_ivp(self.assignmentModelAntitheticX4, 
                                      [self.muTimePoints[i], self.muTimePoints[i+1]], 
                                      self.X0,)
            T = sol.t; X = sol.y
            
            fullX = np.hstack((fullX, X))
            time = np.append(time, T)
            fullMuSig = np.append(fullMuSig, np.repeat(self.muVec[i], len(T)))
            
            self.X0 = X[:,-1]
            
        return fullX.T, time, fullMuSig
        
    def plotFittedModel(self, beta3): # 9
        
        x1, x5, time, fullMuSig = self.runAssignmentModel()
        
        plt.figure(figsize=(7, 6))
        plt.plot(time, x1, time, x5); plt.plot(time, fullMuSig, linestyle="--", color="k")
        plt.scatter(self.timePoints[1:], self.x1Exp, c="tab:blue")
        plt.scatter(self.timePoints[1:], self.x5Exp, c="tab:orange")
        plt.legend(["X1", "X5", "Input", "Experimental X1", "Experimental X5"]);
        plt.xlabel("Time (hr)"); plt.ylabel("Species")
        plt.title("Species levels over Time")
        caption = rf"$\it{{Figure\ {self.figCounter}}}$: $\beta_3$ = {self.beta3:.2f}hr$^{{-1}}$"
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        self.figCounter += 1
        
    def plotDist(self, distOption=1): # 10
        
        match distOption:
            
            case 1: # disturbance 1
                
                x1, x5, time, fullMuSig = self.runAssignmentModel(delta1=0.5)
                caption = rf"$\it{{Figure\ {self.figCounter}}}$: $\Delta_1 = {self.delta1}hr^{{-1}}$, applied at 144hrs."
            
            case 2: # disturbance 2
                
                x1, x5, time, fullMuSig = self.runAssignmentModel(delta2=0.5, delta2_time=144)
                caption = rf"$\it{{Figure\ {self.figCounter}}}$: $\Delta_2 = {self.delta2}hr^{{-1}}$, applied at 144hrs."
        
        plt.figure(figsize=(7, 6))
        plt.plot(time, x1, time, x5); plt.plot(time, fullMuSig, linestyle="--", color="k")
        plt.legend(["X1", "X5", "Input"]);
        plt.xlabel("Time (hr)"); plt.ylabel("Species")
        plt.title(f"Species levels over Time (Disturbance {distOption})")
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        self.figCounter += 1
        
    def plotVaryDist(self, distOption=1): # 11     
        
        delta12Vec = np.array([-0.9, -0.5, 0.5, 1])
        
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 7.5))
        ax = ax.flatten()
        
        match distOption:
            
            case 1: # disturbance 1
                
                fig.suptitle(r"Comparison of Varying Disturbance $\Delta_{1}$")
                
                for i in range(len(delta12Vec)):
                    
                    x1, x5, time, fullMuSig = self.runAssignmentModel(delta1=delta12Vec[i])
                    ax[i].set_title(rf"$\Delta_1 = {self.delta1}hr^{{-1}}$")
                    ax[i].plot(time, x1, time, x5); ax[i].plot(time, fullMuSig, linestyle="--", 
                                                               color="k")
            
            case 2: # disturbance 2
                
                fig.suptitle(r"Comparison of Varying Disturbance $\Delta_{2}$")
                
                for i in range(len(delta12Vec)):
                    
                    x1, x5, time, fullMuSig = self.runAssignmentModel(delta2=delta12Vec[i])
                    ax[i].set_title(rf"$\Delta_2 = {self.delta2}hr^{{-1}}$")
                    ax[i].plot(time, x1, time, x5); ax[i].plot(time, fullMuSig, linestyle="--", 
                                                               color="k")
        
        for i in range(len(ax)):
            
            ax[i].set_ylabel("Species"); ax[i].set_xlabel("Time (hr)")
            ax[i].legend(["X1", "X5", "Input"], loc="upper left");
            
        caption = rf"$\it{{Figure\ {self.figCounter}}}$: The disturbance is always applied at 144hrs."
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        plt.tight_layout()
        self.figCounter += 1
        
    def plotBothDist(self): # 12
        
        delta2_time_vec = [96, ] # make into subplots later?
        
        for delta2_time in delta2_time_vec:
            x1, x5, time, fullMuSig = self.runAssignmentModel(delta1=0.5, delta2=1, delta2_time=delta2_time)
            plt.figure(figsize=(7, 6))
            plt.plot(time, x1, time, x5); plt.plot(time, fullMuSig, linestyle="--", color="k")
            plt.legend(["X1", "X5", "Input"]);
            plt.xlabel("Time (hr)"); plt.ylabel("Species")
            plt.title("Species levels over Time (Disturbance 1 and 2)")
            caption = rf"$\it{{Figure\ {self.figCounter}}}$: $\Delta_1 = {self.delta1}hr^{{-1}}$, $\Delta_2 = {self.delta2}hr^{{-1}}$ ($\Delta_2$ applied at {delta2_time}$hrs$)."
            plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
            self.figCounter += 1
            
    def plotProportionalControl(self, Kp): # 13
        
        self.K = Kp
        
        x1, x5, time, fullMuSig = self.runAssignmentModel(delta1=0.5, delta2=1, delta2_time=96)
        
        plt.figure(figsize=(7, 6))
        plt.plot(time, x1, time, x5); plt.plot(time, fullMuSig, linestyle="--", color="k")
        plt.legend(["X1", "X5", "Input"]);
        plt.xlabel("Time (hr)"); plt.ylabel("Species")
        plt.title("Species levels over Time (Proportional Control)")
        caption = rf"$\it{{Figure\ {self.figCounter}}}$: $K_{{P}}$ = {self.K:.2f}"
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        self.figCounter += 1
        
    def plotIntegralControl(self, theta=1, eta2=0.1): # 14
        
        self.theta = theta
        self.eta2 = eta2
        self.K = 0
        
        x1, x5, time, fullMuSig = self.runAssignmentModelAntithetic(delta1=0.5, delta2=1, delta2_time=96)

        plt.figure(figsize=(7, 6))
        plt.plot(time, x1, time, x5); plt.plot(time, fullMuSig, linestyle="--", color="k")
        plt.legend(["X1", "X5", "Input"]);
        plt.xlabel("Time (hr)"); plt.ylabel("Species")
        plt.title("Species levels over Time (Integral Control)")
        caption = rf"$\it{{Figure\ {self.figCounter}}}$: $K_{{I}}$ = {self.theta:.2f}, $\eta_2$ = {self.eta2:.2f}"
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        self.figCounter += 1
        
    def plotPIControl(self, theta=1, eta2=0.1, Kp=0, dilution=0): # 15
        
        self.theta = theta
        self.eta2 = eta2
        self.K = Kp
        self.dilution = dilution
        
        x1, x5, time, fullMuSig = self.runAssignmentModelAntithetic(delta1=0.5, delta2=1, delta2_time=96)

        plt.figure(figsize=(7, 6))
        plt.plot(time, x1, time, x5); plt.plot(time, fullMuSig, linestyle="--", color="k")
        plt.legend(["X1", "X5", "Input"]);
        plt.xlabel("Time (hr)"); plt.ylabel("Species")
        plt.title("Species levels over Time (Proportional-Integral Control)")
        caption = rf"$\it{{Figure\ {self.figCounter}}}$: $K_{{I}}$ = {self.theta:.2f}, $\eta_2$ = {self.eta2:.2f}, $K_{{P}}$ = {self.K:.2f}, $\lambda$ = {self.dilution}"
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        self.figCounter += 1
        
    def plotAllControlX4(self, theta=[0,0,1,1], eta2=[0,0,0.1,0.1], Kp=[0,0.5,0,0.5]): # 16
        # inputs need to be lists/tuples 
         
        X, time, fullMuSig = self.runAssignmentModelAntitheticX4(delta1=0.5, delta2=1, delta2_time=96)
        titles = ["No Control: ", "Proportional Control: ", "Integral Control: ", "PI Control: "]

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 7.5))
        ax = ax.flatten()
        
        for i in range(len(ax)-2):
        
            self.theta = theta[i]
            self.eta2 = eta2[i]
            self.K = Kp[i]
            
            X, time, fullMuSig = self.runAssignmentModelX4(delta1=0.5, delta2=1, delta2_time=96)
            ax[i].plot(time, X, linewidth=2); ax[i].plot(time, fullMuSig, linestyle="--", color="k")
            ax[i].set_xlabel("Time (hr)"); ax[i].set_ylabel("Species")
            ax[i].set_title(f"{titles[i]}" + rf"$\theta$={self.theta:.2f}, $\eta_{{2}}$={self.eta2:.2f}, $K$={self.K:.2f}")
            ax[i].legend(["Z1", "X1", "X2", "X3", "X4", "X5", "Input"]);
        
        for i in range(2, len(ax)):
        
            self.theta = theta[i]
            self.eta2 = eta2[i]
            self.K = Kp[i]
            
            X, time, fullMuSig = self.runAssignmentModelAntitheticX4(delta1=0.5, delta2=1, delta2_time=96)
            ax[i].plot(time, X, linewidth=2); ax[i].plot(time, fullMuSig, linestyle="--", color="k")
            ax[i].set_xlabel("Time (hr)"); ax[i].set_ylabel("Species")
            ax[i].set_title(f"{titles[i]}" + rf"$\theta$={self.theta:.2f}, $\eta_{{2}}$={self.eta2:.2f}, $K$={self.K:.2f}")
            ax[i].legend(["Z1", "X1", "X2", "X3", "X4", "X5", "Z2", "Input"]);
            
        caption = rf"$\it{{Figure\ {self.figCounter}}}$: A comparison of all three controllers' performance on ensuring $X_{4}$ tracks $\mu$."
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        plt.tight_layout()
        self.figCounter += 1

    def plotAllControlX5(self, theta=[0,0,1,1], eta2=[0,0,0.1,0.1], Kp=[0,0.5,0,0.5], dilution=0): # 16
        # inputs need to be lists
        
        self.dilution = dilution
        
        titles = ["No Control: ", "Proportional Control: ", "Integral Control: ", "PI Control: "]
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7.5, 7.5))
        ax = ax.flatten()
        
        for i in range(len(ax)-2):
        
            self.theta = theta[i]
            self.eta2 = eta2[i]
            self.K = Kp[i]
            
            x1, x5, time, fullMuSig = self.runAssignmentModel(delta1=0.5, delta2=1, delta2_time=96)
            ax[i].plot(time, x1, time, x5, linewidth=2); ax[i].plot(time, fullMuSig, linestyle="--", color="k")
            ax[i].set_xlabel("Time (hr)"); ax[i].set_ylabel("Species")
            ax[i].set_title(f"{titles[i]}" + rf"$\theta$={self.theta:.2f}, $\eta_{{2}}$={self.eta2:.2f}, $K$={self.K:.2f}")
            ax[i].legend(["X1", "X5", "Input"]);
        
        for i in range(2, len(ax)):
        
            self.theta = theta[i]
            self.eta2 = eta2[i]
            self.K = Kp[i]
            
            x1, x5, time, fullMuSig = self.runAssignmentModelAntithetic(delta1=0.5, delta2=1, delta2_time=96)
            ax[i].plot(time, x1, time, x5, linewidth=2); ax[i].plot(time, fullMuSig, linestyle="--", color="k")
            ax[i].set_xlabel("Time (hr)"); ax[i].set_ylabel("Species")
            ax[i].set_title(f"{titles[i]}" + rf"$\theta$={self.theta:.2f}, $\eta_{{2}}$={self.eta2:.2f}, $K$={self.K:.2f}")
            ax[i].legend(["X1", "X5", "Input"]);
            
        caption = rf"$\it{{Figure\ {self.figCounter}}}$: A comparison of all three controllers' performance on ensuring $X_{5}$ tracks $\mu$."
        plt.figtext(0.5, 0.001, caption, wrap=True, horizontalalignment='center', fontsize=10);
        plt.tight_layout()
        self.figCounter += 1
        
    def assignmentModelKpCost(self, Kp): # 17
        # cost function that, when minimised, whill return optimal Kp

        self.X0 = np.ones((6, )) * 1E-09
        self.K = Kp[0]
        
        delta1 = 0.5
        delta2 = 1
        delta2_time = 96
        
        cost = 0
        
        for i in range(len(self.muTimePoints)-1):
            
            self.assignmentModelOptions["mu"] = self.muVec[i]
            
            if self.muTimePoints[i] < 144:
                self.delta1 = 0
            elif self.muTimePoints[i] >= 144:
                self.delta1 = delta1
            
            if self.muTimePoints[i] < delta2_time:
                self.delta2 = 0
            elif self.muTimePoints[i] >= delta2_time:
                self.delta2 = delta2
                
            sol = integrate.solve_ivp(self.assignmentModel, 
                                      [self.muTimePoints[i], self.muTimePoints[i+1]], 
                                      self.X0,)
            T = sol.t; X = sol.y
            x5 = X[5,:]
            
            self.X0 = X[:,-1]
            
            error = self.muVec[i] - x5
            cost += integrate.trapezoid(np.abs(error), T)
            
            if np.any(X < 0):
                
                negPenalty = np.sum(np.minimum(0, X)**2)
                cost += 1e4 * negPenalty
            
        return cost
    
    def assignmentModelKICost(self, integralParams): # 18
        # cost function that, when minimised, whill return optimal KI

        self.X0 = np.ones((7, )) * 1E-09
        self.theta = integralParams[0]
        self.eta2 = integralParams[1]
        self.K = 0
        
        delta1 = 0.5
        delta2 = 1
        delta2_time = 96
        
        cost = 0
        
        for i in range(len(self.muTimePoints)-1):
            
            self.assignmentModelOptions["mu"] = self.muVec[i]
            
            if self.muTimePoints[i] < 144:
                self.delta1 = 0
            elif self.muTimePoints[i] >= 144:
                self.delta1 = delta1
            
            if self.muTimePoints[i] < delta2_time:
                self.delta2 = 0
            elif self.muTimePoints[i] >= delta2_time:
                self.delta2 = delta2
                
            sol = integrate.solve_ivp(self.assignmentModelAntithetic, 
                                      [self.muTimePoints[i], self.muTimePoints[i+1]], 
                                      self.X0,)
            T = sol.t; X = sol.y
            x5 = X[5,:]
            
            self.X0 = X[:,-1]
            
            error = self.muVec[i] - x5
            cost += integrate.trapezoid(np.abs(error), T)
            
            if np.any(X < 0):
                
                negPenalty = np.sum(np.minimum(0, X)**2)
                cost += 1e4 * negPenalty
            
        return cost
    
    def assignmentModelKPICost(self, integralParams, dilution): # 19
        # cost function that, when minimised, whill return optimal KP, KI

        self.X0 = np.ones((7, )) * 1E-09
        self.theta = integralParams[0]
        self.eta2 = integralParams[1]
        self.K = integralParams[2]
        self.dilution = dilution
        
        delta1 = 0.5
        delta2 = 1
        delta2_time = 96
        
        cost = 0
        
        for i in range(len(self.muTimePoints)-1):
            
            self.assignmentModelOptions["mu"] = self.muVec[i]
            
            if self.muTimePoints[i] < 144:
                self.delta1 = 0
            elif self.muTimePoints[i] >= 144:
                self.delta1 = delta1
            
            if self.muTimePoints[i] < delta2_time:
                self.delta2 = 0
            elif self.muTimePoints[i] >= delta2_time:
                self.delta2 = delta2
                
            sol = integrate.solve_ivp(self.assignmentModelAntithetic, 
                                      [self.muTimePoints[i], self.muTimePoints[i+1]], 
                                      self.X0,)
            T = sol.t; X = sol.y
            x5 = X[5,:]
            
            self.X0 = X[:,-1]
            
            error = self.muVec[i] - x5
            cost += integrate.trapezoid(np.abs(error), T)
            
            if np.any(X < 0):
                
                negPenalty = np.sum(np.minimum(0, X)**2)
                cost += 1e4 * negPenalty
            
        return cost