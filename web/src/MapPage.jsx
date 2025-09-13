import { useEffect, useMemo, useState } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, ScaleControl } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

// 後端位址（vite：在 .env 內用 VITE_API_BASE_URL 設定；否則預設 localhost:8000）
const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const CITY = 'TPE' // 目前以台北市為例，如要切換城市可改由 UI 控制

// 產生 24 小時、每 5 分鐘一個選項
function useTimeOptions() {
  return useMemo(() => {
    const out = []
    for (let h = 0; h < 24; h++) {
      for (let m = 0; m < 60; m += 5) {
        const hh = String(h).padStart(2, '0')
        const mm = String(m).padStart(2, '0')
        out.push(`${hh}:${mm}`)
      }
    }
    return out
  }, [])
}

function colorByProba(p) {
  if (p == null || Number.isNaN(p)) return '#95a5a6' // 尚未預測
  if (p >= 0.8) return '#2ecc71'
  if (p >= 0.6) return '#8bc34a'
  if (p >= 0.4) return '#f1c40f'
  if (p >= 0.2) return '#ff9800'
  return '#e74c3c'
}

function joinDateTime(dateStr, timeStr) {
  if (!dateStr || !timeStr) return ''
  return `${dateStr} ${timeStr}`
}

export default function MapPage() {
  const [stations, setStations] = useState([]) // [{sno,sna,lat,lng,tot,available_now?,proba?,pred?}]
  const [loadingStations, setLoadingStations] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')

  // 預設日期：今天；預設時間：現在+30 分
  const now = new Date()
  const pad = n => String(n).padStart(2, '0')
  const [defaultDate] = useState(`${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}`)
  const [defaultTime] = useState(() => {
    const t = new Date(now.getTime() + 30*60000)
    return `${pad(t.getHours())}:${pad(t.getMinutes())}`
  })
  const timeOptions = useTimeOptions()

  // 1) 載入所有站點（只拿基本資訊，不做預測）
  useEffect(() => {
    async function loadStations() {
      try {
        setLoadingStations(true)
        setErrorMsg('')
        const res = await fetch(`${API}/api/stations?city=${CITY}`)
        const arr = await res.json()
        const init = (Array.isArray(arr) ? arr : []).map(s => ({
          ...s,
          proba: null,      // 可租機率（0~1）
          predAvail: null,  // 預估可借數量（整數）
        }))
        setStations(init)
      } catch (err) {
        console.error('[api/stations] failed:', err)
        setErrorMsg('載入站點失敗，請確認 API 是否啟動')
        setStations([])
      } finally {
        setLoadingStations(false)
      }
    }
    loadStations()
  }, [])

  // 2) 單站預測：用戶在彈窗內選日期+時間，送出後更新該站資料
  async function predictOne(sno, dateStr, timeStr) {
    const target = joinDateTime(dateStr, timeStr)
    if (!target) return
    try {
      const url = `${API}/api/predict_one?city=${encodeURIComponent(CITY)}&sno=${encodeURIComponent(sno)}&target=${encodeURIComponent(target)}`
      const res = await fetch(url)
      const d = await res.json()
      console.log('[predict_one result]', d)
      if (d?.ok) {
        setStations(prev => prev.map(s =>
          String(s.sno) === String(sno)
            ? { ...s, proba: d.proba_can_rent ?? null, predAvail: d.pred_available ?? null }
            : s
        ))
      } else {
        alert(`預測失敗：${d?.msg ?? 'unknown error'}`)
      }
    } catch (e) {
      console.error('[api/predict_one] failed:', e)
      alert('呼叫 /api/predict_one 失敗，請查看瀏覽器 Console 或後端日誌')
    }
  }

  return (
    <div style={{ position: 'relative', height: '100vh', width: '100%' }}>
      {/* 右下角圖例 */}
      <div style={{ position:'absolute', right:12, bottom:12, zIndex:1000,
                    background:'#fff', padding:'8px 12px', borderRadius:8,
                    boxShadow:'0 2px 12px rgba(0,0,0,0.15)', fontSize:13 }}>
        <div style={{ fontWeight:600, marginBottom:4 }}>機率圖例</div>
        <div><span style={{display:'inline-block',width:10,height:10,background:'#2ecc71',borderRadius:2,marginRight:6}}></span>≥ 0.8</div>
        <div><span style={{display:'inline-block',width:10,height:10,background:'#8bc34a',borderRadius:2,marginRight:6}}></span>0.6–0.79</div>
        <div><span style={{display:'inline-block',width:10,height:10,background:'#f1c40f',borderRadius:2,marginRight:6}}></span>0.4–0.59</div>
        <div><span style={{display:'inline-block',width:10,height:10,background:'#ff9800',borderRadius:2,marginRight:6}}></span>0.2–0.39</div>
        <div><span style={{display:'inline-block',width:10,height:10,background:'#e74c3c',borderRadius:2,marginRight:6}}></span>{'< 0.2'}</div>
        <div style={{marginTop:6,color:'#666'}}>
          {loadingStations ? '載入站點中…' : (errorMsg || '')}
        </div>
      </div>

      <MapContainer center={[25.04, 121.55]} zoom={12} style={{ height: '100%', width: '100%' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />
        <ScaleControl position="bottomleft" metric />

        {stations.map((s) => {
          const color = colorByProba(s.proba)
          return (
            <CircleMarker
              key={s.sno}
              center={[s.lat, s.lng]}
              radius={7}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.9 }}
            >
              <Popup>
                <PerStationPopup
                  station={stations.find(x => String(x.sno) === String(s.sno)) || s}
                  defaultDate={defaultDate}
                  defaultTime={defaultTime}
                  timeOptions={timeOptions}
                  onPredict={(dateStr, timeStr) => predictOne(s.sno, dateStr, timeStr)}
                />
              </Popup>
            </CircleMarker>
          )
        })}
      </MapContainer>
    </div>
  )
}

// 子元件：單一站點的彈窗內容（含時間輸入與下拉）
function PerStationPopup({ station, defaultDate, defaultTime, timeOptions, onPredict }) {
  const [dateStr, setDateStr] = useState(defaultDate)
  const [timeInput, setTimeInput] = useState(defaultTime)
  const [timeSelect, setTimeSelect] = useState(defaultTime)

  // 兩個時間來源保持同步：使用者改其中一個，就套用到另一個
  useEffect(() => { setTimeSelect(timeInput) }, [timeInput])
  useEffect(() => { setTimeInput(timeSelect) }, [timeSelect])

  const probaPct = station.proba == null ? '-' : `${Math.round(station.proba * 100)}%`
  const availStr = station.predAvail == null ? '-' : `${station.predAvail} 台`

  return (
    <div style={{ minWidth: 280 }}>
      <div style={{ fontWeight: 600 }}>{station.sna ?? station.sno}</div>
      <div style={{ color: '#555', marginBottom: 8 }}>
        目前可借：{station.available_now ?? '-'} / {station.tot ?? '-'}
      </div>

      <div style={{ display:'flex', gap:8, alignItems:'center', marginBottom:6 }}>
        <label style={{ width: 72 }}>日期</label>
        <input type="date" value={dateStr} onChange={e => setDateStr(e.target.value)} />
      </div>

      <div style={{ display:'flex', gap:8, alignItems:'center', marginBottom:6 }}>
        <label style={{ width: 72 }}>時間（輸入）</label>
        <input type="time" value={timeInput} onChange={e => setTimeInput(e.target.value)} step={60} />
      </div>

      <div style={{ display:'flex', gap:8, alignItems:'center', marginBottom:8 }}>
        <label style={{ width: 72 }}>時間（下拉）</label>
        <select value={timeSelect} onChange={e => setTimeSelect(e.target.value)}>
          {timeOptions.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>

      <button
        onClick={() => onPredict(dateStr, timeInput)}
        style={{ padding:'6px 10px', borderRadius:6, background:'#2563eb', color:'#fff', border:'none' }}
      >
        預測此時段
      </button>

      <div style={{ marginTop:8, lineHeight:1.6 }}>
        指定時間：{dateStr} {timeInput}<br/>
        可租機率：<b>{probaPct}</b><br/>
        預估可借：<b>{availStr}</b>
      </div>
    </div>
  )
}
