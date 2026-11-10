// Лабораторная работа №11 — Логика 3D-цеха (Three.js), ML/DL прогнозирования и оптимизации ресурсов

// --- 1. ПАРАМЕТРЫ СИМУЛЯЦИИ И МОДЕЛЕЙ ---
const ANOMALY_THRESHOLD = 0.68; // 95-й процентиль ошибки на валидации (кВт·ч)
const TARIFF_E = 28.0;          // тг / кВт·ч
const TARIFF_W = 0.25;          // тг / л

const MODELS = {
    rf:   { name: "Случайный лес (Random Forest)", mae: 0.2315, rmse: 0.4820, r2: 0.9785, noise: 0.16 },
    lstm: { name: "Глубокое обучение (PyTorch LSTM, L=12)", mae: 0.2580, rmse: 0.5120, r2: 0.9710, noise: 0.19 },
    lr:   { name: "Линейная регрессия", mae: 0.4120, rmse: 0.7410, r2: 0.9410, noise: 0.32 },
    base: { name: "Базовый прогноз E(t+1) = E(t)", mae: 1.1420, rmse: 1.9540, r2: 0.8240, noise: 0.55 }
};

const SCHEDULES = {
    nominal: {
        name: "Режим 1: Исходный (1 линия, 4200 л/ч)",
        flow: 4200, lines: 1, hours: 4.8,
        totalE: 727.20, totalW: 7180.0, ePerL: 0.0364, wPerL: 0.3590, cost: 22156.60
    },
    optimal: {
        name: "Режим 4: Рекомендуемый (1 линия, 3800 л/ч)",
        flow: 3800, lines: 1, hours: 5.3,
        totalE: 694.20, totalW: 7180.0, ePerL: 0.0347, wPerL: 0.3590, cost: 21232.60
    },
    parallel: {
        name: "Режим 3: Две линии (2 x 2500 л/ч)",
        flow: 5000, lines: 2, hours: 4.0,
        totalE: 752.40, totalW: 7360.0, ePerL: 0.0376, wPerL: 0.3680, cost: 22907.20
    },
    forced: {
        name: "Режим 2: Форсированный (1 линия, 5200 л/ч)",
        flow: 5200, lines: 1, hours: 3.9,
        totalE: 768.10, totalW: 7210.0, ePerL: 0.0384, wPerL: 0.3605, cost: 23309.30
    }
};

const EQUIPMENT_INFO = {
    "TANK-01": {
        id: "TANK-01",
        name: "Резервуар сырого молока",
        status: "Работает",
        params: "Объём: 14 200 л | Температура: 4.1 °C",
        action: "Штатное охлаждение и перемешивание сырья"
    },
    "PASTEUR-01": {
        id: "PASTEUR-01",
        name: "Участок тепловой обработки (ПОУ)",
        status: "Работает",
        params: "Температура: 76.5 °C | Давление: 3.1 бар | Выдержка: 20 с",
        action: "Контроль температурного режима пастеризации (>= 76 °C)"
    },
    "COOLER-01": {
        id: "COOLER-01",
        name: "Секция ледяного охлаждения",
        status: "Работает",
        params: "Температура выхода: 4.0 °C | Поток ледяной воды: 1 450 л/ч",
        action: "Рекуперация тепла в норме"
    },
    "PACK-01": {
        id: "PACK-01",
        name: "Фасовочная линия Pure-Pak",
        status: "Работает",
        params: "Скорость: 70 пак/мин | Выпуск смены: 19 850 л",
        action: "Синхронизация подачи с буферным танком"
    },
    "PANEL-01": {
        id: "PANEL-01",
        name: "Центральный пульт оператора АСУ ТП",
        status: "Онлайн",
        params: "Канал телеметрии: JSON / REST API | Шаг: 5 мин",
        action: "Мониторинг невязки r_t и управление расписанием"
    }
};

let state = {
    scenario: "baseline",
    modelKey: "rf",
    scheduleKey: "nominal",
    selectedNode: "PASTEUR-01",
    stepCount: 0,
    anomalyStreak: 0,
    simTime: new Date("2026-10-15T10:00:00"),
    historyLabels: [],
    historyActualE: [],
    historyPredE: [],
    historyResiduals: [],
    lastActualE: 9.8,
    lastPredE: 9.75,
    lastWater: 122.0
};

// --- 2. ИНИЦИАЛИЗАЦИЯ 3D СЦЕНЫ (THREE.JS) ---
let scene, camera, renderer, controls;
let equipmentMeshes = {};
let beaconMeshes = {};
let milkPacks = [];
let flowParticles = [];
let agitatorBlade;

function init3D() {
    const container = document.getElementById("viewport3d");
    const canvas = document.getElementById("canvas3d");

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1120);
    scene.fog = new THREE.FogExp2(0x0b1120, 0.018);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 200);
    camera.position.set(16, 13, 20);

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(0, 2, 0);

    // Освещение
    const ambient = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xe0f2fe, 0.9);
    dirLight.position.set(15, 25, 15);
    dirLight.castShadow = true;
    scene.add(dirLight);

    // Пол и координатная сетка цеха
    const floorGeo = new THREE.PlaneGeometry(42, 26);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.85, metalness: 0.1 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    const grid = new THREE.GridHelper(42, 42, 0x1e3a8a, 0x1e293b);
    grid.position.y = 0.01;
    scene.add(grid);

    const steelMat = new THREE.MeshStandardMaterial({ color: 0xcbd5e1, metalness: 0.75, roughness: 0.25 });
    const blueMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.5, roughness: 0.4 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.6, roughness: 0.5 });

    // 1. TANK-01: Резервуар сырого молока
    const tankGroup = new THREE.Group();
    tankGroup.position.set(-11, 0, 0);
    const tankBody = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.2, 6.2, 32), steelMat.clone());
    tankBody.position.y = 3.6;
    tankBody.castShadow = true;
    tankGroup.add(tankBody);

    const tankTop = new THREE.Mesh(new THREE.ConeGeometry(2.2, 1.2, 32), steelMat);
    tankTop.position.y = 7.3;
    tankGroup.add(tankTop);

    agitatorBlade = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.2, 0.4), blueMat);
    agitatorBlade.position.y = 7.9;
    tankGroup.add(agitatorBlade);

    scene.add(tankGroup);
    equipmentMeshes["TANK-01"] = tankBody;
    beaconMeshes["TANK-01"] = createBeacon(tankGroup, 0, 8.6, 0);

    // 2. PASTEUR-01: Пластинчатая ПОУ и гомогенизатор
    const pasteurGroup = new THREE.Group();
    pasteurGroup.position.set(-3.5, 0, 0);
    const pouBase = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.5, 2.4), darkMat);
    pouBase.position.y = 0.25;
    pasteurGroup.add(pouBase);

    const pouCore = new THREE.Mesh(new THREE.BoxGeometry(2.8, 3.4, 1.8), steelMat.clone());
    pouCore.position.y = 2.2;
    pouCore.castShadow = true;
    pasteurGroup.add(pouCore);

    // Трубчатый выдерживатель
    const holderCoil = new THREE.Mesh(new THREE.TorusGeometry(0.7, 0.14, 16, 32), blueMat);
    holderCoil.position.set(0, 2.5, 1.05);
    pasteurGroup.add(holderCoil);

    scene.add(pasteurGroup);
    equipmentMeshes["PASTEUR-01"] = pouCore;
    beaconMeshes["PASTEUR-01"] = createBeacon(pasteurGroup, 0, 4.5, 0);

    // 3. COOLER-01: Охладитель
    const coolerGroup = new THREE.Group();
    coolerGroup.position.set(3.2, 0, 0);
    const coolerMesh = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.4, 4.2, 24), blueMat.clone());
    coolerMesh.position.y = 2.3;
    coolerMesh.castShadow = true;
    coolerGroup.add(coolerMesh);

    scene.add(coolerGroup);
    equipmentMeshes["COOLER-01"] = coolerMesh;
    beaconMeshes["COOLER-01"] = createBeacon(coolerGroup, 0, 5.0, 0);

    // 4. PACK-01: Линия фасовки Pure-Pak и конвейер
    const packGroup = new THREE.Group();
    packGroup.position.set(10.0, 0, 0);
    const fillerTower = new THREE.Mesh(new THREE.BoxGeometry(2.6, 4.0, 2.4), steelMat.clone());
    fillerTower.position.set(-1.2, 2.0, 0);
    fillerTower.castShadow = true;
    packGroup.add(fillerTower);

    const conveyorBelt = new THREE.Mesh(new THREE.BoxGeometry(7.0, 0.35, 1.2), darkMat);
    conveyorBelt.position.set(1.5, 1.1, 0);
    packGroup.add(conveyorBelt);

    // Пакеты молока на конвейере
    const packMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3 });
    for (let i = 0; i < 6; i++) {
        const carton = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.72, 0.38), packMat);
        carton.position.set(8.8 + i * 1.0, 1.65, 0);
        scene.add(carton);
        milkPacks.push(carton);
    }

    scene.add(packGroup);
    equipmentMeshes["PACK-01"] = fillerTower;
    beaconMeshes["PACK-01"] = createBeacon(packGroup, -1.2, 4.6, 0);

    // 5. PANEL-01: Пульт оператора
    const panelGroup = new THREE.Group();
    panelGroup.position.set(0, 0, 6.2);
    const desk = new THREE.Mesh(new THREE.BoxGeometry(3.2, 1.3, 1.2), darkMat);
    desk.position.y = 0.65;
    panelGroup.add(desk);

    const screen = new THREE.Mesh(
        new THREE.BoxGeometry(2.6, 1.4, 0.15),
        new THREE.MeshBasicMaterial({ color: 0x0284c7 })
    );
    screen.position.set(0, 1.85, -0.2);
    screen.rotation.x = -0.25;
    panelGroup.add(screen);

    scene.add(panelGroup);
    equipmentMeshes["PANEL-01"] = screen;
    beaconMeshes["PANEL-01"] = createBeacon(panelGroup, 0, 3.1, 0);

    // Трубопроводы и частицы потока молока
    createPipelines();

    // Клик мышью по 3D-объектам
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    renderer.domElement.addEventListener("pointerdown", (e) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        const entries = Object.entries(equipmentMeshes);
        for (const [id, mesh] of entries) {
            const intersects = raycaster.intersectObject(mesh, true);
            if (intersects.length > 0) {
                focusEquipment(id);
                break;
            }
        }
    });

    window.addEventListener("resize", () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    animate3D();
}

function createBeacon(parent, x, y, z) {
    const mat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.32, 16, 16), mat);
    sphere.position.set(x, y, z);
    parent.add(sphere);
    return sphere;
}

function createPipelines() {
    const pipeMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.8, roughness: 0.2 });
    const pipeGeo = new THREE.CylinderGeometry(0.14, 0.14, 19.5, 16);
    const mainPipe = new THREE.Mesh(pipeGeo, pipeMat);
    mainPipe.rotation.z = Math.PI / 2;
    mainPipe.position.set(-1.2, 2.8, 0);
    scene.add(mainPipe);

    const pMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
    for (let i = 0; i < 18; i++) {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.19, 10, 10), pMat);
        dot.position.set(-11 + i * 1.15, 2.8, 0);
        scene.add(dot);
        flowParticles.push(dot);
    }
}

function animate3D() {
    requestAnimationFrame(animate3D);
    const isStopped = (state.scenario === "downtime");

    if (!isStopped && agitatorBlade) {
        agitatorBlade.rotation.y += 0.06;
    }

    // Движение пакетов и частиц потока
    if (!isStopped && state.scenario !== "cip") {
        milkPacks.forEach((p) => {
            p.position.x += 0.035;
            if (p.position.x > 14.8) p.position.x = 8.8;
        });
    }

    if (!isStopped) {
        flowParticles.forEach((pt) => {
            pt.position.x += 0.06;
            if (pt.position.x > 9.0) pt.position.x = -11.0;
        });
    }

    // Пульсация аварийного маяка на пастеризаторе
    const pasteurBeacon = beaconMeshes["PASTEUR-01"];
    if (pasteurBeacon) {
        if (state.anomalyStreak >= 3) {
            const pulse = (Math.sin(Date.now() * 0.01) + 1) * 0.5;
            pasteurBeacon.material.color.setHex(pulse > 0.5 ? 0xef4444 : 0x7f1d1d);
            equipmentMeshes["PASTEUR-01"].material.color.setHex(pulse > 0.5 ? 0xef4444 : 0xcbd5e1);
        } else if (state.scenario === "cip") {
            pasteurBeacon.material.color.setHex(0xf59e0b);
            equipmentMeshes["PASTEUR-01"].material.color.setHex(0xcbd5e1);
        } else {
            pasteurBeacon.material.color.setHex(0x22c55e);
            equipmentMeshes["PASTEUR-01"].material.color.setHex(0xcbd5e1);
        }
    }

    controls.update();
    renderer.render(scene, camera);
}

function focusEquipment(id) {
    state.selectedNode = id;
    document.querySelectorAll(".cam-btn").forEach((b) => b.classList.remove("active"));
    const activeBtn = document.querySelector(`[data-cam="${id}"]`);
    if (activeBtn) activeBtn.classList.add("active");

    const targets = {
        "OVERVIEW":   { pos: [16, 13, 20], look: [0, 2, 0] },
        "TANK-01":    { pos: [-11, 7, 11], look: [-11, 3.5, 0] },
        "PASTEUR-01": { pos: [-3.5, 6, 9], look: [-3.5, 2.2, 0] },
        "COOLER-01":  { pos: [3.2, 6, 9],  look: [3.2, 2.2, 0] },
        "PACK-01":    { pos: [10.5, 6, 9], look: [10.5, 2.0, 0] },
        "PANEL-01":   { pos: [0, 4.5, 12], look: [0, 1.5, 6.2] }
    };

    const t = targets[id] || targets["OVERVIEW"];
    camera.position.set(...t.pos);
    controls.target.set(...t.look);

    updateInspectorHUD();
}

// --- 3. ГРАФИКИ TELEMETRY (CHART.JS) ---
let energyChart, residualChart;

function initCharts() {
    const ctx1 = document.getElementById("energyChart").getContext("2d");
    energyChart = new Chart(ctx1, {
        type: "line",
        data: {
            labels: state.historyLabels,
            datasets: [
                {
                    label: "Факт E_t (кВт·ч)",
                    data: state.historyActualE,
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.1)",
                    borderWidth: 2,
                    tension: 0.25,
                    pointRadius: 2
                },
                {
                    label: "Прогноз Ê_{t+1} (кВт·ч)",
                    data: state.historyPredE,
                    borderColor: "#a855f7",
                    borderDash: [4, 4],
                    borderWidth: 2,
                    tension: 0.25,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 10 } } } },
            scales: {
                x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.04)" } },
                y: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { color: "rgba(255,255,255,0.06)" } }
            }
        }
    });

    const ctx2 = document.getElementById("residualChart").getContext("2d");
    residualChart = new Chart(ctx2, {
        type: "bar",
        data: {
            labels: state.historyLabels,
            datasets: [
                {
                    label: "Ошибка r_t = |E_t - Ê_t|",
                    data: state.historyResiduals,
                    backgroundColor: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { labels: { color: "#cbd5e1", font: { size: 10 } } },
                subtitle: {
                    display: true,
                    text: "Порог 95-го процентиля: 0.68 кВт·ч",
                    color: "#f59e0b",
                    font: { size: 10 }
                }
            },
            scales: {
                x: { ticks: { color: "#64748b", font: { size: 9 } }, grid: { display: false } },
                y: { beginAtZero: true, suggestedMax: 1.5, ticks: { color: "#64748b", font: { size: 9 } } }
            }
        }
    });
}

// --- 4. ШАГ СИМУЛЯЦИИ ТЕХПРОЦЕССА И МОДЕЛЕЙ ---
function stepSimulation() {
    state.stepCount++;
    state.simTime = new Date(state.simTime.getTime() + 5 * 60000); // +5 минут
    const timeStr = state.simTime.toTimeString().slice(0, 5);

    const sched = SCHEDULES[state.scheduleKey];
    const model = MODELS[state.modelKey];

    let baseFlow = sched.flow;
    let processTemp = 76.5 + (Math.random() - 0.5) * 0.3;
    let loadPct = Math.round((baseFlow / 5200) * 100);

    // Учёт активного сценария
    if (state.scenario === "high_demand") {
        baseFlow = Math.min(5400, baseFlow * 1.18);
        loadPct = 96;
    } else if (state.scenario === "low_milk") {
        baseFlow = 3300;
        loadPct = 64;
    } else if (state.scenario === "downtime") {
        baseFlow = 0;
        loadPct = 5;
        processTemp = 22.0;
    } else if (state.scenario === "cip") {
        baseFlow = 0;
        loadPct = 30;
        processTemp = 75.0;
    }

    // Ожидаемое потребление энергии по физической модели (за 5 мин интервал)
    const nominalKw = (baseFlow > 0)
        ? (45.0 + 0.015 * baseFlow + 0.6 * (processTemp - 4.0))
        : (state.scenario === "cip" ? 55.0 : 8.0);

    const expectedE = nominalKw * (5 / 60);
    const predE = Math.max(0.2, expectedE + (Math.random() - 0.5) * model.noise);

    // Фактическое потребление (при сценарии аномалии добавляется скачок +3.8 кВт·ч)
    let actualE = expectedE + (Math.random() - 0.5) * 0.22;
    if (state.scenario === "anomaly") {
        actualE += 3.8;
    }

    const waterStep = (state.scenario === "cip")
        ? 250.0 + (Math.random() - 0.5) * 8.0
        : (baseFlow * (5 / 60) * 0.35);

    const residual = Math.abs(actualE - predE);

    if (residual > ANOMALY_THRESHOLD) {
        state.anomalyStreak++;
    } else {
        state.anomalyStreak = 0;
    }

    state.lastActualE = actualE;
    state.lastPredE = predE;
    state.lastWater = waterStep;

    // Обновление массивов графика (окно 18 точек)
    state.historyLabels.push(timeStr);
    state.historyActualE.push( Number(actualE.toFixed(2)) );
    state.historyPredE.push( Number(predE.toFixed(2)) );
    state.historyResiduals.push( Number(residual.toFixed(2)) );

    if (state.historyLabels.length > 18) {
        state.historyLabels.shift();
        state.historyActualE.shift();
        state.historyPredE.shift();
        state.historyResiduals.shift();
    }

    // Цвета столбцов ошибки
    residualChart.data.datasets[0].backgroundColor = state.historyResiduals.map((val) =>
        val > ANOMALY_THRESHOLD ? "#ef4444" : "#22c55e"
    );

    energyChart.update();
    residualChart.update();

    // Обновление UI карточек
    document.getElementById("kpiActualE").textContent = `${actualE.toFixed(2)} кВт·ч`;
    document.getElementById("kpiPredE").textContent = `${predE.toFixed(2)} кВт·ч`;
    document.getElementById("kpiWater").textContent = `${waterStep.toFixed(1)} л`;
    document.getElementById("kpiResidual").textContent = `${residual.toFixed(2)} кВт·ч`;
    document.getElementById("hudTimestamp").textContent = timeStr;
    document.getElementById("hudLoad").textContent = `${loadPct}%`;
    document.getElementById("hudTemp").textContent = `${processTemp.toFixed(1)} °C`;

    // Проверка условия тревоги (несколько интервалов подряд)
    const banner = document.getElementById("alertBanner");
    const topDot = document.getElementById("statusDot");
    const topLabel = document.getElementById("statusText");

    if (state.anomalyStreak >= 3) {
        banner.classList.add("active");
        topDot.className = "dot danger";
        topLabel.textContent = "ОТКЛОНЕНИЕ ЭНЕРГИИ";
    } else if (state.scenario === "cip") {
        banner.classList.remove("active");
        topDot.className = "dot warning";
        topLabel.textContent = "Режим: Мойка CIP";
    } else {
        banner.classList.remove("active");
        topDot.className = "dot";
        topLabel.textContent = "Режим: Штатное производство";
    }

    updateInspectorHUD();
}

function updateInspectorHUD() {
    const info = EQUIPMENT_INFO[state.selectedNode] || EQUIPMENT_INFO["PASTEUR-01"];
    document.getElementById("inspName").textContent = `${info.id} — ${info.name}`;
    document.getElementById("inspParams").textContent = info.params;
    document.getElementById("inspPred").textContent =
        `Энергия (t+1): ${state.lastPredE.toFixed(2)} кВт·ч | Вода: ${state.lastWater.toFixed(1)} л`;

    if (state.anomalyStreak >= 3 && state.selectedNode === "PASTEUR-01") {
        document.getElementById("inspStatus").textContent = "ПРЕДУПРЕЖДЕНИЕ (Аномалия расхода)";
        document.getElementById("inspStatus").style.color = "#ef4444";
        document.getElementById("inspAction").textContent =
            "Обнаружено отклонение потребления энергии. Требуется проверка режима работы оборудования";
    } else {
        document.getElementById("inspStatus").textContent = info.status;
        document.getElementById("inspStatus").style.color = "#22c55e";
        document.getElementById("inspAction").textContent = info.action;
    }
}

function updateScheduleMetrics() {
    const s = SCHEDULES[state.scheduleKey];
    document.getElementById("optCost").textContent = `${s.cost.toLocaleString("ru-RU")} тг`;
    document.getElementById("optEPerL").textContent = `${s.ePerL} кВт·ч/л`;
    document.getElementById("optWPerL").textContent = `${s.wPerL} л/л`;
    const savings = (SCHEDULES.nominal.cost - s.cost).toFixed(0);
    document.getElementById("optSavings").textContent =
        savings > 0 ? `Экономия: +${savings} тг на партии` : `Перерасход относительно оптимума`;
}

// --- 5. ЭКСПОРТ VR_TELEMETRY.JSON ---
function downloadTelemetryJSON() {
    const payload = {
        timestamp: new Date().toISOString(),
        active_scenario: state.scenario,
        selected_ml_model: MODELS[state.modelKey].name,
        equipment: Object.values(EQUIPMENT_INFO).map((eq) => ({
            id: eq.id,
            name: eq.name,
            status: (state.anomalyStreak >= 3 && eq.id === "PASTEUR-01") ? "warning" : "working",
            alert: (state.anomalyStreak >= 3 && eq.id === "PASTEUR-01"),
            alert_message: (state.anomalyStreak >= 3 && eq.id === "PASTEUR-01")
                ? "Обнаружено отклонение потребления энергии. Требуется проверка режима работы оборудования"
                : "OK"
        })),
        predictions: {
            next_interval_electricity_kwh: Number(state.lastPredE.toFixed(2)),
            next_interval_water_l: Number(state.lastWater.toFixed(1)),
            anomaly_threshold_kwh: ANOMALY_THRESHOLD
        },
        optimization: SCHEDULES[state.scheduleKey]
    };

    const blob = new Blob([JSON.stringify(payload, null, 4)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "vr_telemetry.json";
    a.click();
    URL.revokeObjectURL(url);
}

// --- 6. ПРИВЯЗКА СОБЫТИЙ ИНТЕРФЕЙСА ---
window.addEventListener("DOMContentLoaded", () => {
    init3D();
    initCharts();
    updateScheduleMetrics();

    // Стартовая генерация 10 точек истории
    for (let i = 0; i < 10; i++) stepSimulation();

    setInterval(stepSimulation, 2200);

    document.getElementById("scenarioSelect").addEventListener("change", (e) => {
        state.scenario = e.target.value;
        if (state.scenario === "anomaly") {
            focusEquipment("PASTEUR-01");
        }
        stepSimulation();
    });

    document.getElementById("modelSelect").addEventListener("change", (e) => {
        state.modelKey = e.target.value;
        const m = MODELS[state.modelKey];
        document.getElementById("modelMetricsLine").textContent =
            `MAE: ${m.mae} кВт·ч | RMSE: ${m.rmse} кВт·ч | R²: ${m.r2}`;
        stepSimulation();
    });

    document.getElementById("scheduleSelect").addEventListener("change", (e) => {
        state.scheduleKey = e.target.value;
        updateScheduleMetrics();
        stepSimulation();
    });

    document.querySelectorAll(".cam-btn").forEach((btn) => {
        btn.addEventListener("click", () => focusEquipment(btn.dataset.cam));
    });

    document.getElementById("btnExportJson").addEventListener("click", downloadTelemetryJSON);

    // Модальные окна
    document.querySelectorAll("[data-modal]").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.getElementById(btn.dataset.modal).classList.add("open");
        });
    });

    document.querySelectorAll(".close-modal").forEach((btn) => {
        btn.addEventListener("click", () => {
            btn.closest(".modal-backdrop").classList.remove("open");
        });
    });
});
