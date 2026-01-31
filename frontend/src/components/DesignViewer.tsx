import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Stars, Float, Text, ContactShadows, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { PointCloudViewer } from './PointCloudViewer';

interface ComponentProps {
    id: string;
    type: string;
    position: [number, number, number];
    properties?: any;
    name?: string;
}

const PipeModel = ({ start, end, diameter, isSelected }: { start: number[], end: number[], diameter: number, isSelected?: boolean }) => {
    const startVec = new THREE.Vector3(...start);
    const endVec = new THREE.Vector3(...end);
    const direction = new THREE.Vector3().subVectors(endVec, startVec);
    const length = direction.length();
    if (length < 0.1) return null;

    const midpoint = new THREE.Vector3().addVectors(startVec, endVec).multiplyScalar(0.5);
    const up = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction.normalize());

    return (
        <group position={midpoint} quaternion={quaternion}>
            <mesh castShadow receiveShadow>
                <cylinderGeometry args={[diameter / 24, diameter / 24, length, 32]} />
                <meshStandardMaterial
                    color={isSelected ? "#6366f1" : "#475569"}
                    metalness={0.8}
                    roughness={0.2}
                    emissive={isSelected ? "#6366f1" : "#000000"}
                    emissiveIntensity={isSelected ? 0.3 : 0}
                />
            </mesh>
            {/* Flanges at ends */}
            <mesh position={[0, length / 2, 0]}>
                <cylinderGeometry args={[diameter / 12, diameter / 12, 0.1, 32]} />
                <meshStandardMaterial color="#334155" metalness={0.9} />
            </mesh>
            <mesh position={[0, -length / 2, 0]}>
                <cylinderGeometry args={[diameter / 12, diameter / 12, 0.1, 32]} />
                <meshStandardMaterial color="#334155" metalness={0.9} />
            </mesh>
        </group>
    );
};

const PumpModel = ({ isSelected }: { isSelected: boolean }) => (
    <group>
        {/* Motor */}
        <mesh position={[0, 0.4, -0.4]} castShadow rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.4, 0.4, 1, 16]} />
            <meshStandardMaterial color={isSelected ? "#818cf8" : "#334155"} metalness={0.8} />
        </mesh>
        {/* Pump Housing */}
        <mesh position={[0, 0.4, 0.4]} castShadow>
            <cylinderGeometry args={[0.6, 0.6, 0.8, 32]} />
            <meshStandardMaterial color={isSelected ? "#6366f1" : "#0ea5e9"} metalness={0.9} />
        </mesh>
        {/* Base plate */}
        <mesh position={[0, 0.05, 0]} castShadow>
            <boxGeometry args={[1.2, 0.1, 1.8]} />
            <meshStandardMaterial color="#1e293b" />
        </mesh>
    </group>
);

const TankModel = ({ isSelected, properties }: { isSelected: boolean, properties?: any }) => {
    const shape = properties?.shape || 'vertical';

    if (shape === 'spherical') {
        return (
            <mesh castShadow>
                <sphereGeometry args={[4, 64, 64]} />
                <meshStandardMaterial
                    color={isSelected ? "#6366f1" : "#1e293b"}
                    metalness={0.8}
                    roughness={0.2}
                />
            </mesh>
        );
    }

    if (shape === 'rectangular') {
        return (
            <mesh position={[0, 4, 0]} castShadow>
                <boxGeometry args={[8, 8, 8]} />
                <meshStandardMaterial
                    color={isSelected ? "#6366f1" : "#1e293b"}
                    metalness={0.4}
                    roughness={0.6}
                />
            </mesh>
        );
    }

    // Default: Vertical Cylinder
    return (
        <group>
            <mesh position={[0, 4, 0]} castShadow>
                <cylinderGeometry args={[4, 4, 8, 64]} />
                <meshStandardMaterial
                    color={isSelected ? "#6366f1" : "#1e293b"}
                    metalness={0.6}
                    roughness={0.4}
                    envMapIntensity={1}
                />
            </mesh>
            {/* Top Dome */}
            <mesh position={[0, 8, 0]} castShadow>
                <sphereGeometry args={[4, 64, 32, 0, Math.PI * 2, 0, Math.PI / 2]} />
                <meshStandardMaterial color="#334155" metalness={0.8} />
            </mesh>
            {/* Ladder detail */}
            <mesh position={[4.1, 4, 0]}>
                <boxGeometry args={[0.2, 8, 0.8]} />
                <meshStandardMaterial color="#475569" />
            </mesh>
        </group>
    );
};

const ValveModel = ({ isSelected }: { isSelected: boolean }) => (
    <group rotation={[0, 0, Math.PI / 2]}>
        <mesh castShadow>
            <sphereGeometry args={[0.6, 32, 32]} />
            <meshStandardMaterial color={isSelected ? "#6366f1" : "#f43f5e"} />
        </mesh>
        {/* Actuator */}
        <group position={[0, 0.8, 0]}>
            <mesh castShadow>
                <boxGeometry args={[0.4, 0.8, 0.4]} />
                <meshStandardMaterial color="#475569" />
            </mesh>
            <mesh position={[0, 0.5, 0]}>
                <cylinderGeometry args={[0.6, 0.6, 0.2, 32]} />
                <meshStandardMaterial color="#334155" />
            </mesh>
        </group>
    </group>
);

const FilterModel = ({ isSelected }: { isSelected: boolean }) => (
    <group>
        <mesh position={[0, 2, 0]} castShadow>
            <boxGeometry args={[2.5, 4, 2.5]} />
            <meshStandardMaterial color={isSelected ? "#6366f1" : "#f59e0b"} metalness={0.4} roughness={0.6} />
        </mesh>
        {/* Support Legs */}
        {[-1, 1].map(x => [-1, 1].map(z => (
            <mesh key={`${x}-${z}`} position={[x * 1, 0, z * 1]}>
                <cylinderGeometry args={[0.1, 0.1, 1, 8]} />
                <meshStandardMaterial color="#334155" />
            </mesh>
        )))}
    </group>
);

const ModelComponent = ({ type, position, properties, name, isSelected, onSelect }: ComponentProps & { isSelected: boolean, onSelect: () => void }) => {
    if (type === 'pipe' && properties?.start_pos && properties?.end_pos) {
        return <PipeModel start={properties.start_pos} end={properties.end_pos} diameter={properties.diameter || 4} isSelected={isSelected} />;
    }

    return (
        <group position={position} onClick={(e) => { e.stopPropagation(); onSelect(); }}>
            <Float speed={1.5} rotationIntensity={0.1} floatIntensity={0.3}>
                {type === 'pump' && <PumpModel isSelected={isSelected} />}
                {type === 'tank' && <TankModel isSelected={isSelected} properties={properties} />}
                {type === 'valve' && <ValveModel isSelected={isSelected} />}
                {type === 'filter' && <FilterModel isSelected={isSelected} />}
            </Float>

            {/* Component Label */}
            <Text
                position={[0, type === 'tank' ? 9 : type === 'filter' ? 5 : 2.5, 0]}
                fontSize={0.6}
                color={isSelected ? "#818cf8" : "white"}
                anchorX="center"
                anchorY="middle"
                maxWidth={4}
                textAlign="center"
            >
                {name || type.toUpperCase()}
            </Text>
        </group>
    );
};

export const DesignViewer = ({
    components = [],
    onSelectComponent,
    customContent = null,
    pointCloud = null,
    pointQuality = 'high'
}: {
    components: any[],
    onSelectComponent?: (comp: any) => void,
    customContent?: React.ReactNode,
    pointCloud?: Array<{ pos?: [number, number, number]; color?: [number, number, number]; x?: number; y?: number; z?: number; r?: number; g?: number; b?: number }> | null,
    pointQuality?: 'fast' | 'normal' | 'high' | 'ultra'
}) => {
    const [selectedId, setSelectedId] = useState<string | null>(null);

    const handleSelect = (comp: any) => {
        setSelectedId(comp.id);
        if (onSelectComponent) onSelectComponent(comp);
    };

    // Convert point cloud to scaled positions for display
    const scaledPointCloud = pointCloud ? pointCloud.map(p => {
        const x = p.pos ? p.pos[0] : (p.x || 0);
        const y = p.pos ? p.pos[1] : (p.y || 0);
        const z = p.pos ? p.pos[2] : (p.z || 0);
        return {
            ...p,
            x: x * 15,  // Scale up for visibility in the environment
            y: y * 15 + 5,  // Lift above platform
            z: z * 15
        };
    }) : null;

    return (
        <div className="w-full h-full min-h-[600px] rounded-3xl overflow-hidden glass-card relative bg-[#010409]">
            <Canvas shadows dpr={[1, 2]} onPointerMissed={() => { setSelectedId(null); if (onSelectComponent) onSelectComponent(null); }}>
                <PerspectiveCamera makeDefault position={[50, 40, 50]} fov={35} />
                <OrbitControls makeDefault minDistance={10} maxDistance={300} dampingFactor={0.05} />

                <Environment preset="city" />
                <ambientLight intensity={0.2} />
                <spotLight position={[50, 100, 50]} angle={0.2} penumbra={1} intensity={1.5} castShadow shadow-mapSize={2048} />
                <pointLight position={[-30, 20, -30]} intensity={1} color="#38bdf8" />
                <pointLight position={[30, -20, 30]} intensity={0.5} color="#0CEF8F" />

                <Stars radius={150} depth={50} count={3000} factor={4} saturation={0} fade speed={0.5} />
                <Grid
                    infiniteGrid
                    fadeDistance={150}
                    sectionSize={10}
                    sectionColor="#1e293b"
                    cellColor="#0f172a"
                    sectionThickness={1.5}
                    cellThickness={0.8}
                />

                <ContactShadows resolution={1024} scale={150} blur={2.5} opacity={0.4} far={20} color="#000000" />

                <group>
                    {components.length === 0 ? (
                        /* Empty Simulation Environment */
                        <group>
                            {/* Central holographic platform */}
                            <mesh position={[0, 0.1, 0]} receiveShadow>
                                <cylinderGeometry args={[15, 15, 0.2, 64]} />
                                <meshStandardMaterial color="#0a1628" metalness={0.9} roughness={0.1} />
                            </mesh>

                            {/* Glowing ring */}
                            <mesh position={[0, 0.15, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                                <ringGeometry args={[14, 15, 64]} />
                                <meshBasicMaterial color="#38bdf8" transparent opacity={0.6} />
                            </mesh>

                            {/* Holographic projection markers */}
                            {[0, 60, 120, 180, 240, 300].map((angle, i) => (
                                <group key={i} rotation={[0, (angle * Math.PI) / 180, 0]}>
                                    <mesh position={[12, 0.5, 0]}>
                                        <boxGeometry args={[0.3, 1, 0.3]} />
                                        <meshStandardMaterial color="#0ea5e9" emissive="#0ea5e9" emissiveIntensity={0.5} />
                                    </mesh>
                                </group>
                            ))}

                            {/* Floating info text */}
                            <Text
                                position={[0, 5, 0]}
                                fontSize={1.5}
                                color="#38bdf8"
                                anchorX="center"
                                anchorY="middle"
                                maxWidth={20}
                                textAlign="center"
                            >
                                AETHER-GEN
                            </Text>
                            <Text
                                position={[0, 3, 0]}
                                fontSize={0.6}
                                color="#64748b"
                                anchorX="center"
                                anchorY="middle"
                                maxWidth={25}
                                textAlign="center"
                            >
                                Generative Engineering Environment
                            </Text>
                            <Text
                                position={[0, 1.5, 0]}
                                fontSize={0.4}
                                color="#475569"
                                anchorX="center"
                                anchorY="middle"
                                maxWidth={30}
                                textAlign="center"
                            >
                                Enter a design prompt to generate 3D components
                            </Text>
                        </group>
                    ) : (
                        /* Render actual components */
                        components.map((c, i) => (
                            <ModelComponent
                                key={c.id || `comp-${i}`}
                                id={c.id}
                                name={c.name}
                                type={c.type}
                                position={c.position}
                                properties={c.properties}
                                isSelected={selectedId === c.id}
                                onSelect={() => handleSelect(c)}
                            />
                        ))
                    )}
                    {/* Render Point Cloud if provided */}
                    {scaledPointCloud && scaledPointCloud.length > 0 && (
                        <group position={[0, 5, 0]}>
                            <PointCloudViewer points={scaledPointCloud} quality={pointQuality} />
                            {/* Label for point cloud */}
                            <Text
                                position={[0, 12, 0]}
                                fontSize={0.8}
                                color="#38bdf8"
                                anchorX="center"
                                anchorY="middle"
                            >
                                3D Point Cloud Model
                            </Text>
                            <Text
                                position={[0, 11, 0]}
                                fontSize={0.4}
                                color="#64748b"
                                anchorX="center"
                                anchorY="middle"
                            >
                                {scaledPointCloud.length.toLocaleString()} points
                            </Text>
                        </group>
                    )}
                    {customContent}
                </group>
            </Canvas>

            {/* Viewport UI Overlays */}
            <div className="absolute top-8 left-8 flex flex-col gap-4 pointer-events-none">
                <div className="glass-card px-5 py-3 rounded-2xl border-slate-700/50 flex flex-col gap-1 pointer-events-auto">
                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] leading-none">Simulation Environment</span>
                    <span className="text-sm font-bold text-white flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                        Live Engineering Twin
                    </span>
                </div>
            </div>

            <div className="absolute top-8 right-8 flex flex-col gap-2 pointer-events-none text-right">
                <div className="glass-card px-4 py-2 rounded-xl border-slate-800 text-[10px] font-bold text-slate-400 uppercase tracking-widest bg-slate-900/40 backdrop-blur-md">
                    Render Engine: <span className="text-prime-400 ml-1">WebGL 3.0</span>
                </div>
                {selectedId && (
                    <div className="glass-card px-4 py-2 rounded-xl border-prime-500/50 text-[10px] font-bold text-prime-400 uppercase tracking-widest bg-prime-500/10 backdrop-blur-md border animate-in fade-in slide-in-from-right-4 duration-300">
                        Focus: {components.find(c => c.id === selectedId)?.name}
                    </div>
                )}
            </div>

            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-3 glass-card p-2 rounded-2xl border-slate-800/50 backdrop-blur-2xl">
                {['Orbit', 'Pan', 'Zoom'].map((hint) => (
                    <div key={hint} className="px-3 py-1.5 rounded-xl bg-slate-900/50 border border-slate-800 text-[9px] font-black uppercase text-slate-500 tracking-wider">
                        {hint}
                    </div>
                ))}
            </div>
        </div>
    );
};
