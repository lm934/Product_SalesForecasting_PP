# Sales Prediction Using Python


![MasterHead](https://www.leadsquared.com/wp-content/uploads/2021/11/banner-4-3.png)

<font size="1">Image Courtesy: https://www.leadsquared.com/wp-content/uploads/2021/11/banner-4-3.png</font>

Click on the following links to checkout the colab file & video presentation.
- [Article](https://medium.com/p/87d2efd85da7)
- [Video](https://drive.google.com/file/d/1v1dNz0WB5ykNDCO4PsN2xP272mWv3Q25/view?usp=sharing)


---

## Problem Statement

The problem at hand involves predicting future sales for product and service-based businesses. With the dynamic nature of the market, businesses need to anticipate sales figures based on various factors, including advertising expenses, audience targeting, and the choice of advertising platforms.

**Challenges:**

1. **Variable Impact of Advertising:** Understanding how different levels of advertising expenditure influence sales is a complex challenge.
2. **Audience Segmentation:** Identifying and analyzing the diverse audience segments to predict their impact on sales.
3. **Platform Optimization:** Determining the optimal advertising platforms for maximizing sales outcomes.

**Objective:**

Develop a machine learning model using Python to predict future sales. The model should provide actionable insights into the relationships between advertising investments, audience segments, and advertising platforms with sales figures.

**Importance:**

Accurate sales predictions empower businesses to optimize their advertising strategies, allocate resources efficiently, and adapt to changing market dynamics effectively.

**Approach:**

The project will utilize Python's machine learning capabilities, employing regression analysis and predictive modeling techniques to create a robust sales prediction model.

**Outcome:**

The successful execution of this project will result in a reliable tool for businesses, enabling them to make informed decisions, enhance marketing strategies, and achieve better sales forecasting accuracy.

---

## Project Summary

**Objective:**

In this data science project, the goal is to predict future sales for product and service-based businesses. The prediction factors include advertising expenditures, audience segmentation, and advertising platforms.

**Scope:**

Sales prediction is a critical tool for businesses, aiding in decision-making related to advertising costs and strategies. The project employs Python for machine learning to forecast future sales.

**Key Components:**

- **Advertising Expenditure Analysis:** Explore the impact of advertising costs on sales predictions.
- **Audience Segmentation:** Understand how different audience segments contribute to variations in sales.
- **Platform Influence:** Investigate the role of advertising platforms in shaping sales outcomes.

**Implementation:**

Python will be the primary tool for implementing machine learning models. Techniques such as regression analysis and predictive modeling will be employed.

**Outcome:**

The project aims to deliver a robust sales prediction model, enabling businesses to make data-driven decisions, optimize advertising strategies, and enhance operational efficiency.

---

## Results

I have selected r2 score as the primary evaluation metric for the Sales Prediction model. And after removing the overfitted models which have r2 scores for train as 100%, we get the final list:

| Sl. No. | Regression Model      |   Train R2 (%) |   Test R2 (%) |
|:--------|:--------------------------|---------------:|--------------:|
|    1    | Linear regression       |       88.98  |      90.99 |
|    2    | Linear regression tuned       |       88.98  |      90.99 |
|    3    | Lasso regression               |       81.82 |      83.65 |
|    4    | Lasso with alpha = 0.01         |       88.98 |      91.02 |
|    5    | Ridge regression         |       88.98 |      90.93 |
|    6    | Ridge with alpha = 1         |       88.98 |      90.93 |
|    7    | Decision tree tuned         |       87.06 |      82.02 |
|    8    | Random forest         |       99.70 |      97.93 |
|    9    | Random forest tuned         |       84.24 |      84.11 |
|    10    | Gradient Boosting Regressor         |       99.87 |      98.17 |
|    11    | Gradient Boosting Regressor Tuned         |       99.58 |      95.11 |

## Conclusion

In the dynamic landscape of product and service-based businesses, the ability to forecast sales is paramount. This project, undertaken during the data science internship at Oasis Infobyte, delved into the realm of sales prediction using machine learning with Python. Let's encapsulate the key findings:

**Insights and Observations:**

1. Sales exhibit a positive correlation with both TV and Radio advertising expenses, signifying the effectiveness of these channels.
2. Notably, there's a strong correlation between TV advertising expenses and sales, emphasizing the impact of TV advertising on driving sales.
3. The R2 score, chosen as the evaluation metric, showcased the accuracy of the model in predicting sales.
4. The Gradient Boosting model emerged as the preferred choice, achieving an impressive 99% training accuracy and 98% testing accuracy.

**Key Takeaways:**

- Understanding the correlation between advertising expenses and sales aids in strategic decision-making.
- The selected model demonstrates robust predictive capabilities, laying the groundwork for effective sales planning.
- The R2 score provides a reliable indicator of the model's accuracy in forecasting sales trends.

This project not only addresses the nuances of sales prediction but also highlights the pivotal role of data science in optimizing business strategies. The findings contribute to informed decision-making, offering valuable insights for future sales initiatives.

---

## Author

- [Arindam Paul](https://www.linkedin.com/in/arindam-paul-19a085187/)

---

## Reference
 - [Oasis Infobyte](https://oasisinfobyte.com/)
