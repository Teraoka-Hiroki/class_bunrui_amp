# 以下を「app.py」に書き込み
token = "AE/DA9enVyvhM3Y2SANsMCTLZKg9gTKmv23" # ご自身のトークンを入力
import streamlit as st
import numpy as np
import pandas as pd
import math
from base64 import b64encode

# タイトルの表示
st.title("「生徒のクラス分け」アプリ")
st.write("量子アニーリングマシン：Fixstars Amplify")

def process_uploaded_file(file):
    df, column11_data, column12_data,column13_data,column14_data,column15_datacolumn2_data, column3_data, column4_data = None, None, None, None, None,None,None,None
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(file)

        # 列ごとにデータをリストに格納
        column11_data = df.iloc[:, 0].tolist()
        column12_data = df.iloc[:, 1].tolist()
        column13_data = df.iloc[:, 2].tolist()
        column14_data = df.iloc[:, 3].tolist()
        column15_data = df.iloc[:, 4].tolist()
        column2_data  = df.iloc[:, 5].tolist()
        column3_data  = df.iloc[:, 6].tolist()
        column4_data  = df.iloc[:, 7].tolist()

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    return df, column11_data,column12_data,column13_data,column14_data,column15_data, column2_data, column3_data, column4_data

def upload_file_youin():
    st.write("生徒の属性ファイルのアップロード")
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

    if uploaded_file is not None:
        # アップロードされたファイルを処理
        with st.spinner("ファイルを処理中..."):
            df, column11_data,column12_data,column13_data,column14_data,column15_data, column2_data, column3_data, column4_data = process_uploaded_file(uploaded_file)

        # アップロードが成功しているか確認
        if df is not None:
            # アップロードされたCSVファイルの内容を表示
            st.write("アップロードされたCSVファイルの内容:")
            st.write(df)
            w11=column11_data
            w12=column12_data
            w13=column13_data
            w14=column14_data
            w15=column15_data
            w1=column2_data
            w2=column3_data
            p=column4_data
            return w11, w12, w13, w14, w15, w1, w2, p

try:
    w11, w12, w13, w14, w15,  w1, w2, p = upload_file_youin()
    st.write("生徒数：N = ",len(w11))
    N=len(w11)

except Exception as e:
    # エラーが発生したときの処理
    st.error("CSVファイルをアップロード後に処理されます".format(e))


try:
# プルダウンメニューで1から15までの整数を選択
    selected_number = st.selectbox("クラス数を入力してください", list(range(1, 16)))

    # ボタンが押されるまで待機
    submit_button = st.button("確定したら押してください")

    # ボタンが押されたら以下のコードが実行される
    if submit_button:
        # 入力フィールドで選択された整数値を入力
        K = st.number_input("選択したクラス数：", min_value=1, max_value=15, value=selected_number)
except Exception as e:
    st.error("生徒数入力後に処理されます".format(e))


def download_csv(data, filename='data.csv'):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=True)

    b64 = b64encode(csv.encode()).decode()
    st.markdown(f'''
    <a href="data:file/csv;base64,{b64}" download="{filename}">
        クラス分け結果のダウンロード
    </a>
    ''', unsafe_allow_html=True)

# 決定変数の作成
from amplify import BinarySymbolGenerator, BinaryPoly

gen = BinarySymbolGenerator()  # 変数のジェネレータを宣言
x = gen.array(N, K)  # 決定変数を作成

num_unique = len(set(p))

# Create a zero matrix of size (length of list, number of unique elements)
one_hot = [[0 for _ in range(num_unique)] for _ in range(len(p))]

# For each element in the list, set the corresponding element in the one-hot matrix to 1
for i, element in enumerate(p):
    one_hot[i][element] = 1

p=np.array(one_hot)
# Print the one-hot matrix
#print(p)

lam1 = 10
lam2 = 10
a11=1
a12=1
a13=1
a14=1
a15=1
b=1
c=1
d=10

cost11  = 1/K * sum((sum(w11[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w11[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))
cost12  = 1/K * sum((sum(w12[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w12[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))
cost13  = 1/K * sum((sum(w13[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w13[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))
cost14  = 1/K * sum((sum(w14[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w14[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))
cost15  = 1/K * sum((sum(w15[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w15[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))

cost2 = 1/K * sum((sum(w1[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w1[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))
cost3 = 1/K * sum((sum(w2[i]*x[i,k] for i in range(N)) - 1/K * sum(sum(w2[i]*x[i,k] for i in range(N)) for k in range(K)))**2 for k in range(K))

#同じクラスだと加算する
cost4_in=0
for k in range(K):
#  for i in range(N):
#    if p[i,k]==1:
#      cost4_in += (sum(p[i,k]*x[i,k] for k in range(K)))**2
    cost4_in = sum((sum(p[i, j] * x[i, j] for j in range(K)))**2 for i in range(N) if p[i, k] == 1)
cost4 = 1/N*cost4_in

cost = a11*cost11 + a12*cost12 + a13*cost13 + a14*cost14 + a15*cost15 + b*cost2 + c*cost3 + d*cost4

penalty1 = lam1 * sum((sum(x[i,k] for k in range(K)) -1 )**2 for i in range(N))
penalty2 = lam2 * sum((sum(x[i,k] for i in range(N)) -N/K )**2 for k in range(K))
penalty = penalty1 + penalty2

y = cost + penalty
moku = y

    ##########
    # 求解
    ##########
import amplify
from amplify.client import FixstarsClient
from amplify import Solver

    # 実行マシンクライアントの設定
client = FixstarsClient()
client.token = token
client.parameters.timeout = 1 * 500  # タイムアウト1秒

    # アニーリングマシンの実行
solver = Solver(client)  # ソルバーに使用するクライアントを設定
result = solver.solve(moku)  # 問題を入力してマシンを実行

    # 解の存在の確認
if len(result) == 0:
    raise RuntimeError("The given constraints are not satisfied")

    ################
    # 結果の取得
    ################
values = result[0].values  # 解を格納
x_solutions = x.decode(values)
sample_array = x_solutions
st.write("結果表示:")
st.write(x_solutions)

# ダウンロードボタンを表示
download_csv(x_solutions)
st.write('')
st.write('')

#生徒の成績テーブルの平均
Wu = 1/K * sum(w11[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('成績1：'f'ave={Wu}')
Wu = 1/K * sum(w12[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('成績2：'f'ave={Wu}')
Wu = 1/K * sum(w13[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('成績3：'f'ave={Wu}')
Wu = 1/K * sum(w14[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('成績4：'f'ave={Wu}')
Wu = 1/K * sum(w15[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('成績5：'f'ave={Wu}')

W1u = 1/K * sum(w1[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('性別：'f'ave1={W1u}')
W2u = 1/K * sum(w2[i]*sum(sample_array[i][k] for k in range(K)) for i in range(N))
st.write('要支援：'f'ave2={W2u}')
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')
st.write('')
#各クラスでの成績1合計、コスト（分散）、標準偏差を表示
st.write('各クラスでの成績1合計、コスト（分散）、標準偏差を表示')
cost = 0
for k in range(K):
  value = 0
  for i in range(N):
    value = value + sample_array[i][k] * w11[i]
  st.write(f'{value=}')
  cost = cost + (value - Wu)**2
cost = 1/K * cost
st.write(f'{cost=}')
standard_deviation = math.sqrt(cost)#標準偏差
st.write(f'{standard_deviation=}')
st.write('')
#各クラスに対して置くべき生徒を表示
for k in range(K):
  st.write(f'{k=}', end=' : ')

  output_text = "    ".join([str(w11[i]) for i in range(N) if sample_array[i][k] == 1])
  st.write(output_text)
  st.write('')#改行
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')
#各クラスでの性別合計、コスト（分散）、標準偏差を表示
st.write('各クラスでの性別合計、コスト（分散）、標準偏差を表示')
cost1 = 0
for k in range(K):
  value1 = 0
  for i in range(N):
    value1 = value1 + sample_array[i][k] * w1[i]
  st.write(f'{value1=}')
  cost1 = cost1 + (value - W1u)**2
cost1 = 1/K * cost1
st.write(f'{cost1=}')
standard_deviation1 = math.sqrt(cost1)#標準偏差
st.write(f'{standard_deviation1=}')
st.write('')
#各クラスに対して置くべき生徒を表示
for k in range(K):
  st.write(f'{k=}', end=' : ')
  output_text = "    ".join([str(w1[i]) for i in range(N) if sample_array[i][k] == 1])
  st.write(output_text)
  st.write('')#改行
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')
#各クラスでの要支援合計、コスト（分散）、標準偏差を表示
st.write('各クラスでの要支援合計、コスト（分散）、標準偏差を表示')
cost2 = 0
for k in range(K):
  value2 = 0
  for i in range(N):
    value2 = value2 + sample_array[i][k] * w2[i]
  st.write(f'{value2=}')
  cost2 = cost2 + (value2 - W2u)**2
cost2 = 1/K * cost2
st.write(f'{cost2=}')
standard_deviation2 = math.sqrt(cost2)#標準偏差
st.write(f'{standard_deviation2=}')
st.write('')
#各クラスに対して置くべき生徒を表示
for k in range(K):
  st.write(f'{k=}', end=' : ')
  output_text = "    ".join([str(w2[i]) for i in range(N) if sample_array[i][k] == 1])
  st.write(output_text)
  st.write('')#改行
st.write('ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー')

#罰金項のチェック
st.write('生徒一人のクラスの確認：count', end='')
for i in range(N):
  count = 0
  for k in range(K):
      count = count + sample_array[i][k]
output_text = "    ".join([str(count) for i in range(N)])
st.write(output_text)

#except Exception as e:
#    st.error("クラス数確定後に計算されます".format(e))
