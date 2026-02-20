import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import type { State, Node } from '../types';

// Fix for Leaflet default icon not appearing in webpack/vite environments
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

interface MapPanelProps {
    state: State | null;
    mode: "real" | "virtual";
    onNodeClick: (nodeId: number) => void;
}

// Helper component to center map on nodes
const MapUpdater: React.FC<{ nodes: Node[] }> = ({ nodes }) => {
    const map = useMap();
    useEffect(() => {
        if (nodes.length > 0) {
            const bounds = nodes.map(n => [n.lat, n.lon] as [number, number]);
            map.fitBounds(bounds, { padding: [50, 50] });
        }
    }, [nodes, map]);
    return null;
};

const MapPanel: React.FC<MapPanelProps> = ({ state, mode, onNodeClick }) => {
    const [imgTimestamp, setImgTimestamp] = useState(Date.now());

    // Refresh image when state changes (specifically path or nodes)
    useEffect(() => {
        setImgTimestamp(Date.now());
    }, [state?.current_path, state?.nodes]);

    // Helper to format time (assuming cost is in seconds and start time is 15:00:00)
    // Export this or pass it down if needed elsewhere, but for now we duplicate or keep it local
    const formatTime = (seconds: number) => {
        const startHour = 15;
        const totalSeconds = Math.floor(seconds) + startHour * 3600;
        
        const h = Math.floor(totalSeconds / 3600) % 24;
        const m = Math.floor((totalSeconds % 3600) / 60);
        const s = totalSeconds % 60;
        
        return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    };

    if (!state) return <div className="flex items-center justify-center h-full bg-gray-100">Loading State...</div>;

    const { nodes, current_path } = state;
    
    // Get path coordinates for Real Map
    const pathCoords = current_path.map(id => {
        const node = nodes.find(n => n.id === id);
        return node ? [node.lat, node.lon] : null;
    }).filter(p => p !== null) as [number, number][];

    if (mode === 'real') {
        return (
            <div className="h-full w-full relative">
                <MapContainer 
                    center={[52.5200, 13.4050]} // Berlin Center
                    zoom={11} 
                    style={{ height: '100%', width: '100%' }}
                >
                    <TileLayer
                        url="https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"
                        subdomains={['1', '2', '3', '4']}
                        attribution='&copy; <a href="https://www.amap.com/">高德地图</a>'
                    />
                    <MapUpdater nodes={nodes} />
                    
                    {/* Path */}
                    <Polyline positions={pathCoords} color="blue" weight={3} />

                    {/* Nodes */}
                    {nodes.map(node => {
                        const isDepot = node.type === 'depot';
                        const isVisited = current_path.includes(node.id);
                        
                        return (
                            <CircleMarker
                                key={node.id}
                                center={[node.lat, node.lon]}
                                radius={isDepot ? 8 : 6}
                                pathOptions={{
                                    color: isDepot ? '#ef4444' : (isVisited ? '#9ca3af' : '#22c55e'), // Tailwind colors
                                    fillColor: isDepot ? '#ef4444' : (isVisited ? '#9ca3af' : '#22c55e'),
                                    fillOpacity: 0.8
                                }}
                                eventHandlers={{
                                    click: () => {
                                        if (!isVisited) onNodeClick(node.id);
                                    }
                                }}
                            >
                                <Tooltip>
                                    <div className="text-sm font-semibold">
                                        Node {node.id} ({node.type})
                                    </div>
                                    <div className="text-xs">
                                        Demand: {node.demand}
                                    </div>
                                    {node.time_window && (
                                        <div className="text-xs text-indigo-600 font-medium">
                                            TW: {formatTime(node.time_window[0])} - {formatTime(node.time_window[1])}
                                        </div>
                                    )}
                                </Tooltip>
                            </CircleMarker>
                        );
                    })}
                </MapContainer>
                <div className="absolute top-4 right-4 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-md z-[1000] border border-gray-200">
                    <span className="text-xs font-semibold text-gray-700 uppercase tracking-wider">Real World Map</span>
                </div>
            </div>
        );
    } else {
        // Virtual Mode (Image from Backend)
        return (
            <div className="h-full w-full flex items-center justify-center bg-gray-900 relative">
                <div className="relative bg-black border border-gray-700 shadow-2xl rounded-lg overflow-hidden" style={{ width: '80%', aspectRatio: '1/1' }}>
                    <img 
                        src={`/api/virtual-map?t=${imgTimestamp}`} 
                        alt="Virtual Map View" 
                        className="w-full h-full object-contain pixelated"
                    />
                </div>
                <div className="absolute top-4 right-4 bg-gray-800/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-md border border-gray-700">
                     <span className="text-xs font-semibold text-white uppercase tracking-wider">Virtual View (224x224)</span>
                </div>
                <div className="absolute bottom-6 bg-black/50 px-4 py-2 rounded-full text-gray-300 text-xs backdrop-blur-sm">
                    Visualizes the VLM input tensor
                </div>
            </div>
        );
    }
};

export default MapPanel;
