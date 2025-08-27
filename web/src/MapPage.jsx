import { useEffect, useRef, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, ScaleControl, useMapEvents } from 'react-leaflet'
import L from 'leaflet'

// ========== 工具：Nominatim 地理編碼（地址 -> [lat,lng]） ==========
async function geocodeAddress(query) {
  if (!query) return null
  const url = new URL('https://nominatim.openstreetmap.org/search')
  url.searchParams.set('q', query)
  url.searchParams.set('format', 'json')
  url.searchParams.set('addressdetails', '1')
  url.searchParams.set('limit', '1')
  const res = await fetch(url, { headers: { 'Accept': 'application/json', 'User-Agent': 'youbike-availability-demo' }})
  const data = await res.json()
  if (!Array.isArray(data) || data.length === 0) return null
  return [parseFloat(data[0].lat), parseFloat(data[0].lon)]
}

// ========== 路由元件（Leaflet Routing Machine） ==========
function Routing({ start, end, profile }) {
  const map = useMap()
  const controlRef = useRef(null)

  useEffect(() => {
    if (!start || !end) {
      if (controlRef.current) {
        map.removeControl(controlRef.current)
        controlRef.current = null
      }
      return
    }

    const serviceUrl = 'https://router.project-osrm.org/route/v1' // OSRM 公共 demo
    const profiles = { walk: 'foot', bike: 'bike', car: 'car' }
    const osrmProfile = profiles[profile] ?? 'foot'

    if (controlRef.current) {
      map.removeControl(controlRef.current)
      controlRef.current = null
    }

    const control = L.Routing.control({
      waypoints: [ L.latLng(start[0], start[1]), L.latLng(end[0], end[1]) ],
      router: L.Routing.osrmv1({ serviceUrl, profile: osrmProfile }),
      lineOptions: { styles: [{ color: '#2c7be5', weight: 5, opacity: 0.8 }] },
      show: false,
      addWaypoints: false,
      fitSelectedRoutes: true,
      routeWhileDragging: false
    }).addTo(map)

    controlRef.current = control
    return () => { if (controlRef.current) map.removeControl(controlRef.current) }
  }, [map, start, end, profile])

  return null
}

// ========== 點擊地圖設定起/終點 ==========
function ClickToSet({ setStart, setEnd, mode }) {
  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng
      if (mode === 'start') setStart([lat, lng])
      if (mode === 'end') setEnd([lat, lng])
    }
  })
  return null
}

// ========== 右上角圖例（沿用你現有的） ==========
function Legend() {
  const map = useMap()
  useEffect(() => {
    const legend = L.control({ position: 'topright' })
    legend.onAdd = () => {
      const div = L.DomUtil.create('div', 'info legend')
      div.innerHTML = `
        <h4>可租機率</h4>
        <i style="background:#2ecc71"></i> 高 (≥ 66%)<br>
        <i style="background:#f1c40f"></i> 中 (33% - 66%)<br>
        <i style="background:#e74c3c"></i> 低 (&lt; 33%)<br>
      `
      return div
    }
    legend.addTo(map)
    return () => legend.remove()
  }, [map])
  return null
}

// ========== 假資料（你的站點資料可替換） ==========
const DEMO = [
  { sno:'D1', sna:'示範站 A', lat:25.0418, lng:121.5366, tot:28, available_now:10, pred_available:8 },
  { sno:'D2', sna:'示範站 B', lat:25.0524, lng:121.5442, tot:22, available_now: 3, pred_available:5 },
  { sno:'D3', sna:'示範站 C', lat:25.0339, lng:121.5654, tot:30, available_now:25, pred_available:24 },
]

export default function MapPage() {
  const [stations, setStations] = useState([])
  const [start, setStart] = useState(null)      // [lat,lng]
  const [end, setEnd] = useState(null)          // [lat,lng]
  const [profile, setProfile] = useState('walk')// 'walk' | 'bike' | 'car'
  const [clickMode, setClickMode] = useState('start') // 點圖設定起點或終點

  // 載入站點（可改為呼叫你的 /api/predict_batch）
  useEffect(() => {
    const url = import.meta.env.VITE_API_BASE_URL &&
                `${import.meta.env.VITE_API_BASE_URL}/api/predict_batch?city=NTP&h=5`
    if (!url) { setStations(DEMO); return }
    fetch(url).then(r => r.json()).then(d => Array.isArray(d) ? setStations(d) : setStations(DEMO)).catch(() => setStations(DEMO))
  }, [])

  // 簡單 UI：搜尋欄 + 模式切換 + 清除
  async function handleGeocode(which) {
    const q = prompt(which === 'start' ? '輸入起點地址' : '輸入終點地址')
    if (!q) return
    const pos = await geocodeAddress(q)
    if (!pos) { alert('找不到這個地址'); return }
    if (which === 'start') setStart(pos)
    else setEnd(pos)
  }

  return (
    <>
      {/* 浮動控制列 */}
      <div style={{
        position:'absolute', zIndex:1000, top:12, left:12,
        background:'white', padding:'8px 12px', borderRadius:8,
        boxShadow:'0 2px 12px rgba(0,0,0,0.15)', display:'flex', gap:8, alignItems:'center'
      }}>
        <button onClick={() => handleGeocode('start')}>輸入起點地址</button>
        <button onClick={() => handleGeocode('end')}>輸入終點地址</button>
        <select value={profile} onChange={e => setProfile(e.target.value)}>
          <option value="walk">步行</option>
          <option value="bike">自行車</option>
          <option value="car">汽車</option>
        </select>
        <select value={clickMode} onChange={e => setClickMode(e.target.value)}>
          <option value="start">點圖設定起點</option>
          <option value="end">點圖設定終點</option>
        </select>
        <button onClick={() => { setStart(null); setEnd(null); }}>清除路線</button>
      </div>

      <MapContainer center={[25.04, 121.55]} zoom={12} style={{ height: '100vh', width: '100%' }}>
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                   attribution="&copy; OpenStreetMap contributors" />
        <ScaleControl position="bottomleft" metric />

        <Legend />
        <ClickToSet setStart={setStart} setEnd={setEnd} mode={clickMode} />

        {/* 起終點標記（可用 CircleMarker 或 Marker） */}
        {start && (
          <CircleMarker center={start} radius={9} pathOptions={{ color:'#2c7be5', fillColor:'#2c7be5', fillOpacity:0.9 }}>
            <Popup>起點</Popup>
          </CircleMarker>
        )}
        {end && (
          <CircleMarker center={end} radius={9} pathOptions={{ color:'#e83e8c', fillColor:'#e83e8c', fillOpacity:0.9 }}>
            <Popup>終點</Popup>
          </CircleMarker>
        )}

        {/* 站點（示範） */}
        {Array.isArray(stations) && stations.map(s => {
          const p = s.prob_rentable ?? (s.pred_available / Math.max(1, s.tot))
          const color = p >= 0.66 ? '#2ecc71' : p >= 0.33 ? '#f1c40f' : '#e74c3c'
          return (
            <CircleMarker key={s.sno} center={[s.lat, s.lng]} radius={7}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.85 }}>
              <Popup>
                <div style={{ minWidth: 180 }}>
                  <strong>{s.sna}</strong><br/>
                  目前可借：{s.available_now ?? '-'} / {s.tot}<br/>
                  5 分鐘後預測：{s.pred_available?.toFixed(1) ?? '-'}<br/>
                  可租機率：約 {(p*100).toFixed(0)}%
                </div>
              </Popup>
            </CircleMarker>
          )
        })}

        {/* 路徑繪製 */}
        <Routing start={start} end={end} profile={profile} />
      </MapContainer>
    </>
  )
}
