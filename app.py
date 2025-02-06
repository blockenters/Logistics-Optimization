import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import platform
import seaborn as sns
from datetime import datetime

import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def fontRegistered():
    font_dirs = [os.getcwd() + '/custom_fonts']
    font_files = fm.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    fm._load_fontmanager(try_read_cache=False)

# 페이지 기본 설정
st.set_page_config(
    page_title="물류 경로 최적화 대시보드",
    page_icon="🚚",
    layout="wide"
)

# 폰트 등록
fontRegistered()
plt.rc('font', family='NanumGothic')

# 제목
st.title("🚚 물류 경로 최적화 대시보드")

# 앱 설명
st.markdown("""
### 📋 앱 소개
이 대시보드는 물류 배송 경로를 최적화하여 효율적인 배송 계획을 수립하는 데 도움을 주는 도구입니다.

#### 🎯 주요 기능
- **최적 경로 탐색**: 다익스트라(Dijkstra) 알고리즘을 사용하여 출발지에서 도착지까지의 최적 경로를 찾습니다.
- **복합 가중치 적용**: 거리와 우선순위를 모두 고려하여 최적의 경로를 결정합니다.
  - 거리가 짧을수록, 우선순위가 높을수록 선호되는 경로로 계산됩니다.
- **시각적 분석**: 네트워크 그래프와 통계 차트를 통해 경로 정보를 한눈에 파악할 수 있습니다.

#### 💡 사용 방법
1. 왼쪽 사이드바에서 출발지와 도착지를 선택합니다.
2. '경로 계산' 버튼을 클릭하면 최적 경로와 관련 통계가 표시됩니다.
---
""")

# 데이터 로드 함수 수정
@st.cache_data
def load_data():
    return pd.read_csv('data/logistics_route_data.csv')

# 그래프 생성 함수
def create_graph(df):
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        origin = row['origin_warehouse']
        destination = row['destination_region']
        distance = row['distance_km']
        priority = row['priority_level']
        weight = distance / priority
        G.add_edge(origin, destination, weight=weight, distance=distance, priority=priority)
    return G

# 최적 경로 탐색 함수
def find_optimal_route(graph, source, target):
    try:
        path = nx.dijkstra_path(graph, source, target, weight='weight')
        total_distance = sum(graph[u][v]['distance'] for u, v in zip(path[:-1], path[1:]))
        total_priority = sum(graph[u][v]['priority'] for u, v in zip(path[:-1], path[1:]))
        return {
            'route': path,
            'total_distance_km': total_distance,
            'total_priority': total_priority
        }
    except nx.NetworkXNoPath:
        return {
            'route': None,
            'total_distance_km': None,
            'total_priority': None
        }

# 경로 시각화 함수
def visualize_route(graph, route_info):
    if not route_info['route']:
        st.warning("시각화할 경로가 없습니다.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)

    # 모든 노드와 엣지 그리기
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightgrey')
    
    # 폰트 설정
    font_family = 'Malgun Gothic' if platform.system() == 'Windows' else 'NanumGothic'
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family=font_family)
    
    nx.draw_networkx_edges(graph, pos, edge_color='lightgrey', arrows=True)

    # 최적 경로 강조 표시
    optimal_edges = list(zip(route_info['route'][:-1], route_info['route'][1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=optimal_edges, edge_color='red', width=3, arrows=True)
    
    plt.title(f"최적 경로 시각화: {' - '.join(route_info['route'])}", fontsize=14)
    plt.axis('off')
    
    return fig

# 메인 애플리케이션
def main():
    # 데이터 로드
    df = load_data()
    G = create_graph(df)
    
    # 전체 최적 경로 계산
    all_routes = []
    for idx, row in df.iterrows():
        route = find_optimal_route(G, row['origin_warehouse'], row['destination_region'])
        if route['route']:  # None이 아닌 경우만 추가
            all_routes.append(route)
    
    # 사이드바 - 경로 선택
    st.sidebar.header("🎯 경로 설정")
    
    # 출발지/도착지 선택
    unique_origins = df['origin_warehouse'].unique()
    unique_destinations = df['destination_region'].unique()
    
    source = st.sidebar.selectbox(
        "출발지 선택",
        options=unique_origins
    )
    
    destination = st.sidebar.selectbox(
        "도착지 선택",
        options=unique_destinations
    )
    
    # 경로 계산 버튼
    if st.sidebar.button("경로 계산"):
        # 최적 경로 찾기
        optimal_route = find_optimal_route(G, source, destination)
        
        # 결과 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 경로 정보")
            st.markdown("*선택하신 출발지에서 도착지까지의 최적 경로와 상세 정보를 확인하실 수 있습니다.*")
            if optimal_route['route']:
                st.success(f"✅ 최적 경로를 찾았습니다!")
                st.write(f"🚚 경로: {' ➞ '.join(optimal_route['route'])}")
                st.write(f"📏 총 거리: {optimal_route['total_distance_km']:.2f} km")
                st.write(f"⭐ 총 우선순위 점수: {optimal_route['total_priority']}")
            else:
                st.error("❌ 경로를 찾을 수 없습니다.")
        
        with col2:
            st.subheader("🗺 경로 시각화")
            fig = visualize_route(G, optimal_route)
            st.pyplot(fig)
        
        # 통계 시각화
        st.subheader("📈 전체 네트워크 통계")
        col3, col4 = st.columns(2)
        
        with col3:
            fig_dist = plt.figure(figsize=(8, 5))
            distances = [d['distance'] for (u, v, d) in G.edges(data=True)]
            sns.histplot(distances, bins=20)
            plt.title('전체 배송 거리 분포')
            plt.xlabel('거리 (km)')
            st.pyplot(fig_dist)
            
            # 거리 분포 해석
            mean_dist = np.mean(distances)
            max_dist = np.max(distances)
            st.markdown(f"""
            **📊 거리 분포 분석**
            - 평균 배송 거리: {mean_dist:.1f}km
            - 최대 배송 거리: {max_dist:.1f}km
            - 대부분의 배송 경로가 {np.percentile(distances, 25):.1f}km ~ {np.percentile(distances, 75):.1f}km 구간에 분포
            """)
        
        with col4:
            fig_priority = plt.figure(figsize=(8, 5))
            priorities = [d['priority'] for (u, v, d) in G.edges(data=True)]
            sns.countplot(x=priorities)
            plt.title('전체 우선순위 분포')
            plt.xlabel('우선순위 레벨')
            st.pyplot(fig_priority)
            
            # 우선순위 분포 해석
            priority_counts = pd.Series(priorities).value_counts().sort_index()
            most_common = priority_counts.idxmax()
            st.markdown(f"""
            **⭐ 우선순위 분포 분석**
            - 가장 많은 우선순위 레벨: {most_common}
            - 우선순위 1(최우선): {priority_counts.get(1, 0)}건 - 긴급배송, 당일배송 등 최우선 처리가 필요한 배송
            - 우선순위 3(일반): {priority_counts.get(3, 0)}건 - 일반적인 배송 일정
            - 우선순위 5(여유): {priority_counts.get(5, 0)}건 - 여유있게 배송 가능한 일반 화물
            
            *우선순위가 낮을수록(1에 가까울수록) 더 중요하고 긴급한 배송을 의미합니다.*
            """)
        
        # 최적화된 경로 통계 시각화
        st.subheader("📊 최적화된 경로 통계")
        col5, col6 = st.columns(2)
        
        with col5:
            fig_opt_dist = plt.figure(figsize=(8, 5))
            opt_distances = [route['total_distance_km'] for route in all_routes]
            sns.histplot(opt_distances, bins=20)
            plt.title('최적화된 경로 거리 분포')
            plt.xlabel('총 거리 (km)')
            st.pyplot(fig_opt_dist)
            
            # 최적화된 거리 분포 해석
            mean_opt_dist = np.mean(opt_distances)
            max_opt_dist = np.max(opt_distances)
            st.markdown(f"""
            **🎯 최적화된 거리 분석**
            - 평균 최적 경로 거리: {mean_opt_dist:.1f}km
            - 최장 최적 경로: {max_opt_dist:.1f}km
            - 최적화를 통해 전체 평균 대비 {((mean_dist - mean_opt_dist) / mean_dist * 100):.1f}% 거리 단축
            """)
        
        with col6:
            fig_opt_priority = plt.figure(figsize=(8, 5))
            opt_priorities = [route['total_priority'] for route in all_routes]
            sns.countplot(x=opt_priorities)
            plt.title('최적화된 경로 우선순위 분포')
            plt.xlabel('총 우선순위 점수')
            st.pyplot(fig_opt_priority)
            
            # 최적화된 우선순위 분포 해석
            mean_opt_priority = np.mean(opt_priorities)
            st.markdown(f"""
            **💫 최적화된 우선순위 분석**
            - 평균 우선순위 점수: {mean_opt_priority:.1f}
            - 최적화된 경로의 {(sum(p <= 3 for p in opt_priorities) / len(opt_priorities) * 100):.1f}%가 우선순위 3 이하로 처리 (긴급 및 일반 배송)
            - 우선순위가 높은 배송(1-2)이 전체의 {(sum(p <= 2 for p in opt_priorities) / len(opt_priorities) * 100):.1f}%를 차지 (최우선 처리 배송)
            
            *우선순위 점수가 낮을수록 더 빠른 배송이 필요한 경로임을 나타냅니다.*
            """)

    st.sidebar.markdown("---")  # 구분선 추가
    
    # 사이드바 - 데이터 설명
    st.sidebar.header("📦 데이터 설명")
    st.sidebar.markdown("""
    - **출발지(origin_warehouse)**: A, B, C 등의 물류 창고
    - **도착지(destination_region)**: 서울시 각 구(강남구, 송파구 등)
    - **거리(distance_km)**: 출발지에서 도착지까지의 거리(km)
    - **우선순위(priority_level)**: 배송 우선순위 (1: 최우선, 3: 일반, 5: 여유)
    """)

    # 전체 배송 요청 분석 섹션 추가
    st.subheader("📊 전체 배송 요청 분석")
    col7, col8 = st.columns(2)

    with col7:
        # 물류센터별 배송 건수
        plt.figure(figsize=(10, 6))
        warehouse_counts = df['origin_warehouse'].value_counts()
        sns.barplot(x=warehouse_counts.index, y=warehouse_counts.values)
        plt.title('물류센터별 배송 건수')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        
        # 물류센터별 분석 설명
        st.markdown(f"""
        **🏭 물류센터별 배송 현황**
        - 가장 많은 배송: {warehouse_counts.index[0]} ({warehouse_counts.values[0]}건)
        - 가장 적은 배송: {warehouse_counts.index[-1]} ({warehouse_counts.values[-1]}건)
        - 평균 배송 건수: {warehouse_counts.mean():.1f}건
        """)

    with col8:
        # 시간대별 배송 건수
        plt.figure(figsize=(10, 6))
        df['hour'] = pd.to_datetime(df['request_time']).dt.hour
        hourly_counts = df['hour'].value_counts().sort_index()
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values)
        plt.title('시간대별 배송 요청 건수')
        plt.xlabel('시간')
        plt.ylabel('건수')
        st.pyplot(plt)
        
        # 시간대별 분석 설명
        peak_hour = hourly_counts.idxmax()
        st.markdown(f"""
        **⏰ 시간대별 배송 현황**
        - 최다 배송 시간대: {peak_hour}시 ({hourly_counts[peak_hour]}건)
        - 피크 시간(5건 이상): {', '.join(f'{h}시' for h in hourly_counts[hourly_counts >= 5].index)}
        - 심야 배송(22-06시) 비율: {(df['hour'].between(22, 23) | df['hour'].between(0, 5)).mean()*100:.1f}%
        """)

    # 배송 지역별 분석
    st.subheader("🗺 배송 지역별 분석")
    col9, col10 = st.columns(2)

    with col9:
        # 목적지별 배송 건수
        plt.figure(figsize=(10, 6))
        dest_counts = df['destination_region'].value_counts()
        sns.barplot(x=dest_counts.index, y=dest_counts.values)
        plt.title('목적지별 배송 건수')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        
        # 지역별 분석 설명
        st.markdown(f"""
        **🏭 배송 지역 현황**
        - 최다 배송 지역: {dest_counts.index[0]} ({dest_counts.values[0]}건)
        - 최소 배송 지역: {dest_counts.index[-1]} ({dest_counts.values[-1]}건)
        - 장거리 배송(300km 이상) 비율: {(df['distance_km'] >= 300).mean()*100:.1f}%
        """)

    with col10:
        # 우선순위별 평균 거리
        plt.figure(figsize=(10, 6))
        priority_dist = df.groupby('priority_level')['distance_km'].mean()
        sns.barplot(x=priority_dist.index, y=priority_dist.values)
        plt.title('우선순위별 평균 배송 거리')
        plt.xlabel('우선순위')
        plt.ylabel('평균 거리 (km)')
        st.pyplot(plt)
        
        # 우선순위별 분석 설명
        st.markdown(f"""
        **🎯 우선순위별 배송 특성**
        - 우선순위 1 평균 거리: {df[df['priority_level']==1]['distance_km'].mean():.1f}km
        - 우선순위 2 평균 거리: {df[df['priority_level']==2]['distance_km'].mean():.1f}km
        - 우선순위 3 평균 거리: {df[df['priority_level']==3]['distance_km'].mean():.1f}km
        
        *우선순위가 높을수록(1에 가까울수록) 평균 배송 거리가 {priority_dist.iloc[0] > priority_dist.iloc[-1] and "길어지는" or "짧아지는"} 경향이 있습니다.*
        """)

if __name__ == "__main__":
    main()
