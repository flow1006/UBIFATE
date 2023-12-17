import pandas as pd
import random


class IntelligentCarInsuranceDataGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def generate_data_provider_dataset(self, output_file):
        data = []
        for _ in range(self.num_samples):
            age = random.randint(18, 65)
            driving_experience = random.randint(1, 20)
            vehicle_type = random.choice(['Sedan', 'SUV', 'Truck'])
            speeding_incidents = random.randint(0, 5)
            coverage_type = random.choice(['Basic', 'Advanced', 'Premium'])
            claims_history = random.choices([0, 1], weights=[0.9, 0.1])[0]

            # Assume InsurancePremium is influenced by various factors
            insurance_premium = age * 0.01 + speeding_incidents * 0.02 + (claims_history * 0.1 if age < 25 else 0)

            data.append([
                age,
                driving_experience,
                vehicle_type,
                speeding_incidents,
                coverage_type,
                claims_history,
                insurance_premium
            ])

        df = pd.DataFrame(
            data,
            columns=[
                'Age',
                'DrivingExperience',
                'VehicleType',
                'SpeedingIncidents',
                'CoverageType',
                'ClaimsHistory',
                'InsurancePremium'
            ]
        )
        df.to_csv(output_file, index=False)
        print(f"已生成包含 {self.num_samples} 个样本的 {output_file} 文件")

    def generate_data_buyer_dataset(self, output_file):
        data = []
        for _ in range(self.num_samples):
            age = random.randint(18, 65)
            driving_experience = random.randint(1, 20)
            vehicle_type = random.choice(['Sedan', 'SUV', 'Truck'])
            speeding_incidents = random.randint(0, 5)
            coverage_type = random.choice(['Basic', 'Advanced', 'Premium'])

            data.append([
                age,
                driving_experience,
                vehicle_type,
                speeding_incidents,
                coverage_type,
            ])

        df = pd.DataFrame(
            data,
            columns=[
                'Age',
                'DrivingExperience',
                'VehicleType',
                'SpeedingIncidents',
                'CoverageType',
            ]
        )
        df.to_csv(output_file, index=False)
        print(f"已生成包含 {self.num_samples} 个样本的 {output_file} 文件")


# 示例使用方式
num_samples = 10000  # 数据样本数量

data_generator = IntelligentCarInsuranceDataGenerator(num_samples)

# 生成数据提供方的虚拟数据集
data_generator.generate_data_provider_dataset('./provider_data.csv')

# 生成数据购买方的虚拟数据集
data_generator.generate_data_buyer_dataset('./buyer_data.csv')
