import pandas as pd 
class BayesianLearning:
    
    def __init__(self, train_data = None):
        if(train_data is not None):
            self.train_data = train_data
            self.dependency_matrix = pd.DataFrame(0, index=list(train_data.columns), columns=list(train_data.columns), dtype=float)
            self.rows = list(train_data.index)
            self.columns = list(train_data.columns)
            self.instances = len(self.rows)
            self.train()
        return 
    
    def train(self, train_data = None):
        if(train_data is not None):
            self.__init__(train_data)
            return
        
        for i in range(1,self.instances):
            
            mapped_fluctuations = {0: [], 1: [], -1: []}
            for j in range(len(self.columns)):
                difference = self.train_data[self.columns[j]][self.rows[i]] - self.train_data[self.columns[j]][self.rows[i-1]]
                if( difference == 1):
                    mapped_fluctuations[1].append(self.columns[j])
                elif(difference == -1):
                    mapped_fluctuations[-1].append(self.columns[j])
                else:
                    mapped_fluctuations[0].append(self.columns[j])
                    
            
            for j in range(len(mapped_fluctuations[1])):
                for k in range(len(mapped_fluctuations[1])):
                    self.dependency_matrix[mapped_fluctuations[1][j]][mapped_fluctuations[1][k]] += 1
            for j in range(len(mapped_fluctuations[1])):
                for k in range(len(mapped_fluctuations[-1])):
                    self.dependency_matrix[mapped_fluctuations[1][j]][mapped_fluctuations[-1][k]] -= 1
            for j in range(len(mapped_fluctuations[-1])):
                for k in range(len(mapped_fluctuations[1])):
                    self.dependency_matrix[mapped_fluctuations[-1][j]][mapped_fluctuations[1][k]] -= 1
            for j in range(len(mapped_fluctuations[-1])):
                for k in range(len(mapped_fluctuations[-1])):
                    self.dependency_matrix[mapped_fluctuations[-1][j]][mapped_fluctuations[-1][k]] += 1
            
        for i in self.dependency_matrix.columns:
            self.dependency_matrix[i] /= self.dependency_matrix[i][i]
        return 
    
    def predict(self, test_data):
        predicted_data = pd.DataFrame(0, index=list(test_data.index), columns=list(test_data.columns))
        
        values_attributes = {}
        for i in range(len(self.train_data.columns)):
            values_attributes[str(self.train_data.columns[i])] = sum(self.train_data[self.train_data.columns[i]])
            
        for i in range(len(list(predicted_data.columns))):
            for k in range(len(predicted_data)):
                predicted_value = 0
                for j in range(len(list(self.dependency_matrix.columns))):
                    predicted_value += (self.dependency_matrix[predicted_data.columns[i]][self.dependency_matrix.columns[j]])*(values_attributes[self.dependency_matrix.columns[j]])
                predicted_value /= (len(self.train_data)*sum(self.dependency_matrix[predicted_data.columns[i]]))
                if(predicted_value >= (values_attributes[predicted_data.columns[i]]/len(self.train_data))):
                    predicted_value = 1
                else:
                    predicted_value = 0
                values_attributes[predicted_data.columns[i]] += predicted_value
                values_attributes[predicted_data.columns[i]] -= self.train_data[predicted_data.columns[i]][k]
                predicted_data[predicted_data.columns[i]][k] = predicted_value
        
        return predicted_data
   
    def save_dependencies(self, name, location = './'):
        self.dependency_matrix.to_csv(location+name)
        return 