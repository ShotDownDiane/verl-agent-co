import axios from 'axios';
import type { State } from './types';

// Use relative path for proxy
const API_URL = '/api'; 

export const resetEnv = async () => {
    const res = await axios.post<State>(`${API_URL}/reset`);
    return res.data;
};

export const step = async (nodeId: number) => {
    const res = await axios.post<State>(`${API_URL}/step`, { node_id: nodeId });
    return res.data;
};

export const getState = async () => {
    const res = await axios.get<State>(`${API_URL}/state`);
    return res.data;
};

export const predict = async (modelName: string) => {
    const res = await axios.post<{
        observation: string;
        thought: string;
        decision: string;
        node_id: number;
    }>(`${API_URL}/predict`, { model_name: modelName });
    return res.data;
};
