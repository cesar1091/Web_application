import streamlit as st
import pandas as pd
import numpy as np
#import pyodbc

def app():
    st.write('---')
    st.write("""
    # Simple Data Sell Protisa

    Shown are the sells per month

    """)
    #DATABASE CONECTION
    #sql_conn = pyodbc.connect('DRIVER={SQL Server};SERVER=51.222.82.146;DATABASE=STRATEGIO_OLAP_PROTISA;UID=Cesar_VS;PWD=Invernalia!2193;Trusted_Connection=no')
    #query = "SELECT d.Region,c.Categoria,c.Marca,c.Segmento,(CASE WHEN DATEPART(DAY, a.CodigoFecha) BETWEEN 1 AND 15 THEN 1 ELSE 2 END) AS Quincena,DATEPART(MONTH, a.CodigoFecha) AS MES,DATEPART(YEAR, a.CodigoFecha) AS AÑO,SUM(a.VENTASINIGV) AS VSIGV FROM [STRATEGIO_OLAP_PROTISA].[pbix].[Ventas] AS a LEFT JOIN [STRATEGIO_OLAP_PROTISA].[pbix].[Producto] AS c ON a.CodigoProductoDistribuidor = c.CodigoProducto LEFT JOIN [STRATEGIO_OLAP_PROTISA].[pbix].[Distribuidor] AS d ON a.CodigoDistribuidor = d.CodigoDistribuidor LEFT JOIN [STRATEGIO_OLAP_PROTISA].[pbix].[Cliente] AS e ON a.CodigoCliente = e.CodigoCliente WHERE DATEPART(YEAR, a.CodigoFecha)>=2017 AND DATEPART(YEAR, a.CodigoFecha)<=2021 AND a.CodigoDistribuidor not in ('20100239559.0','20100239559.1','20100239559.2','20100239559.3','20100239559.7','20100239559.9') AND c.Marca not in ('Ego','Ideal','Sussy') AND a.CodigoDistribuidor IS NOT NULL AND d.Canal NOT IN ('Farmacia') GROUP BY d.Region,c.Categoria,c.Marca,c.Segmento,(CASE WHEN DATEPART(DAY, a.CodigoFecha) BETWEEN 1 AND 15 THEN 1 ELSE 2 END),DATEPART(MONTH, a.CodigoFecha),DATEPART(YEAR, a.CodigoFecha)"
    #dataset = pd.read_sql(query,sql_conn)
    dataset = pd.read_csv('vsigv_quincena.csv')
    #MAKE FORM DATABASE
    def makeform(df):
        df['Quincena'] = df['Quincena'] + 2*(df['MES']-1)
        df = df.drop('MES',axis=1)
        data = pd.pivot_table(data=df,index=['Region', 'Categoria', 'Marca', 'Segmento', 'Quincena'],columns=["AÑO"],values=["VSIGV"])
        data = data.reset_index()
        values = data.values
        data = pd.DataFrame(data=values,columns = ['Region', 'Categoria', 'Marca', 'Segmento', 'Quincena','VSIGV2017','VSIGV2018','VSIGV2019','VSIGV2020','VSIGV2021'])
        data.fillna(0.00,inplace=True)
        valid_set = data[data['VSIGV2021']>=0].drop(['VSIGV2017','VSIGV2018'],axis=1)
        valid_set['Year'] = '2021'
        a = valid_set.values
        columnas = valid_set.columns
        data_train_test = data.drop(["VSIGV2017","VSIGV2021"],axis=1)
        data_train_test['Year'] = '2020'
        b = data_train_test.values
        data_train_test = data.drop(["VSIGV2020","VSIGV2021"],axis=1)
        data_train_test['Year'] = '2019'
        c = data_train_test.values
        d = np.concatenate((a, b, c))
        data_train = pd.DataFrame(d)
        data_train.columns = columnas
        data_train.rename(columns={'VSIGV2019': 'VSIGV2YA',
                                   'VSIGV2020': 'VSIGV1YA',
                                   'VSIGV2021': 'VSIGV'}, inplace=True)
        return data_train

    data_form = makeform(dataset)
    data_form['Quincena'] = data_form['Quincena'].astype('int64')
    data_form['VSIGV'] = data_form['VSIGV'].astype('int64')
    data_form['VSIGV1YA'] = data_form['VSIGV1YA'].astype('int64')
    data_form['VSIGV2YA'] = data_form['VSIGV2YA'].astype('int64')

    #FILTERS
    region = data_form.Region.unique()
    region_choice = st.sidebar.selectbox('Select region:', region)
    categoria = data_form["Categoria"].loc[data_form["Region"]==region_choice].unique()
    categoria_choice = st.sidebar.selectbox('Select category:', categoria)
    marca = data_form["Marca"].loc[(data_form["Region"]==region_choice)&(data_form["Categoria"]==categoria_choice)].unique()
    marca_choice = st.sidebar.selectbox('Select brand:', marca)
    segmento = data_form["Segmento"].loc[(data_form["Region"]==region_choice) &(data_form["Categoria"]==categoria_choice) & (data_form["Marca"]==marca_choice)].unique()
    segmento_choice = st.sidebar.selectbox('Select segment:', segmento)

    #FILTER DATA
    data_form = data_form.loc[(data_form['Region']==region_choice) & (data_form['Categoria']==categoria_choice) & (data_form['Marca']==marca_choice) & (data_form['Segmento']==segmento_choice)]
    #SHOW DATAFRAME
    st.dataframe(data_form)
    #SPLIT DATA
    data_1 = data_form[data_form['VSIGV']>0].drop('Year',axis=1)
    data_2 = data_form[(data_form['VSIGV']>=0) & (data_form['Year']=='2021')].drop('Year',axis=1)

    data_train_dummies_1 = pd.get_dummies(data_1,columns=['Region', 'Categoria', 'Marca', 'Segmento', 'Quincena'],dtype=float)
    data_train_dummies_2 = pd.get_dummies(data_2,columns=['Region', 'Categoria', 'Marca', 'Segmento', 'Quincena'],dtype=float)

    X_1 = data_train_dummies_1.drop(["VSIGV"],axis=1)
    y_1 = data_train_dummies_1[["VSIGV"]]

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X_1,y_1, test_size=0.15, random_state=10079)

    # Fitting Random Forest Regression to the dataset
    # import the regressor
    from sklearn.ensemble import RandomForestRegressor
    # create regressor object
    regressor = RandomForestRegressor(random_state = 0)
    # fit the regressor with x and y data
    regressor.fit(x_train, y_train)
    #UNDUMMIFY FUNCTION
    def undummify(df, prefix_sep="_"):
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col).astype(float)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep)[1])
                    .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df


    X_2 = data_train_dummies_2.drop(["VSIGV"],axis=1)
    predict = regressor.predict(X_2)

    import matplotlib.ticker as ticker
    data_train_undummy = undummify(data_train_dummies_2, prefix_sep="_")
    data_train_undummy['Predict'] = predict
    data_train_undummy['Predict'] = data_train_undummy['Predict'].astype('int64')
    data_train_undummy['Quincena'] = data_train_undummy['Quincena'].astype('int64')
    data_train_undummy['VSIGV'] = data_train_undummy['VSIGV'].astype('int64')
    data_train_undummy['VSIGV1YA'] = data_train_undummy['VSIGV1YA'].astype('int64')
    data_train_undummy['VSIGV2YA'] = data_train_undummy['VSIGV2YA'].astype('int64')
    data_train_undummy = data_train_undummy.sort_values(by=['Quincena'],ascending=True)


    st.write("""
    # Proyeccion de ventas por quincena
    """)
    proyeccion = data_train_undummy.sort_values(by=['Quincena'],ascending=True)
    proyeccion['Año'] = 2021
    st.dataframe(proyeccion[["Quincena","Año","VSIGV","Predict"]])

    def get_table_download_link_csv(df):
        import pybase64
        #csv = df.to_csv(index=False)
        csv = df.to_csv().encode()
        #b64 = base64.b64encode(csv.encode()).decode()
        b64 = pybase64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="captura.csv" target="_blank">Download csv file</a>'
        return href

    st.markdown(get_table_download_link_csv(proyeccion[["Quincena","Año","VSIGV","Predict"]]), unsafe_allow_html=True)

    import plotly.express as px

    fig = px.line(proyeccion[["Quincena","Año","VSIGV","Predict"]], x="Quincena", y=["VSIGV","Predict"])

    st.plotly_chart(fig)
