import { useCallback, useEffect, useRef, useState } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './components/ui/card'
import { Button } from './components/ui/button'
import { Badge } from './components/ui/badge'
import { AlertTriangle, CheckCircle, Camera, History, RefreshCw, Settings2, SlidersHorizontal, UploadCloud, Video, X } from 'lucide-react'
import emailjs from '@emailjs/browser'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'
const EMAILJS_SERVICE_ID = import.meta.env.VITE_EMAILJS_SERVICE_ID ?? ''
const EMAILJS_TEMPLATE_ID = import.meta.env.VITE_EMAILJS_TEMPLATE_ID ?? ''
const EMAILJS_PUBLIC_KEY = import.meta.env.VITE_EMAILJS_PUBLIC_KEY ?? ''
const EMAIL_ALERTS_ENABLED = Boolean(EMAILJS_SERVICE_ID && EMAILJS_TEMPLATE_ID && EMAILJS_PUBLIC_KEY)

if (import.meta.env.DEV) {
  console.log('EmailJS config check', {
    EMAILJS_SERVICE_ID,
    EMAILJS_TEMPLATE_ID,
    EMAILJS_PUBLIC_KEY,
    EMAIL_ALERTS_ENABLED,
  })
}
const MIN_CLASSIFICATION_INTERVAL = 5
const MAX_CLASSIFICATION_INTERVAL = 60 * 60 * 24
const DEFAULT_CLASSIFICATION_INTERVAL = 5 * 60
const PH_LOCAL_NUMBER_LENGTH = 10
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

type CameraDevice = {
  deviceId: string
  label: string
}

type HistoryEntry = {
  id?: number | string
  prediction?: string | null
  status?: string
  alert?: boolean
  source?: string | null
  filename?: string | null
  logged_at?: string | null
  created_at?: string | null
  model_path?: string | null
  file_size?: number | null
  authority_number?: string | null
  authority_email?: string | null
}

type HistoryFilter = 'all' | 'camera' | 'upload'

const HISTORY_LIMIT = 25
const HISTORY_FILTER_LABELS: Record<HistoryFilter, string> = {
  all: 'All Entries',
  camera: 'Live Camera',
  upload: 'Manual Uploads',
}

const formatIntervalLabel = (seconds: number) => {
  if (seconds < 60) {
    return `Every ${seconds} second${seconds === 1 ? '' : 's'}`
  }

  if (seconds < 3600) {
    const minutes = seconds / 60
    const text = Number.isInteger(minutes) ? minutes.toString() : minutes.toFixed(1)
    return `Every ${text} minute${minutes === 1 ? '' : 's'}`
  }

  const hours = seconds / 3600
  if (hours <= 24) {
    const text = Number.isInteger(hours) ? hours.toString() : hours.toFixed(1)
    return `Every ${text} hour${hours === 1 ? '' : 's'}`
  }

  const days = seconds / 86400
  const text = Number.isInteger(days) ? days.toString() : days.toFixed(1)
  return `Every ${text} day${days === 1 ? '' : 's'}`
}

const clampInterval = (value: number) => {
  if (Number.isNaN(value)) return MIN_CLASSIFICATION_INTERVAL
  return Math.min(MAX_CLASSIFICATION_INTERVAL, Math.max(MIN_CLASSIFICATION_INTERVAL, value))
}

function App() {
  const [status, setStatus] = useState<'Normal' | 'Warning'>('Normal')
  const [backendOnline, setBackendOnline] = useState(true)
  const [prediction, setPrediction] = useState('No Growth')
  const [lastChecked, setLastChecked] = useState(new Date())
  const [loading, setLoading] = useState(false)
  const [imageSrc, setImageSrc] = useState<string | null>(null)
  const [mode, setMode] = useState<'upload' | 'camera'>('upload')
  const [isCameraRunning, setIsCameraRunning] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [cameraDevices, setCameraDevices] = useState<CameraDevice[]>([])
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null)
  const [devicesLoading, setDevicesLoading] = useState(false)
  const [deviceLoadError, setDeviceLoadError] = useState<string | null>(null)
  const [classificationIntervalSeconds, setClassificationIntervalSeconds] = useState(DEFAULT_CLASSIFICATION_INTERVAL)
  const [intervalInputValue, setIntervalInputValue] = useState(String(DEFAULT_CLASSIFICATION_INTERVAL))
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [historyEntries, setHistoryEntries] = useState<HistoryEntry[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [historyFilter, setHistoryFilter] = useState<HistoryFilter>('all')
  const [authorityNumber, setAuthorityNumber] = useState('')
  const [authorityNumberError, setAuthorityNumberError] = useState<string | null>(null)
  const [authorityEmail, setAuthorityEmail] = useState('')
  const [authorityEmailError, setAuthorityEmailError] = useState<string | null>(null)
  const [authorityLoading, setAuthorityLoading] = useState(true)
  const [authorityLoadError, setAuthorityLoadError] = useState<string | null>(null)
  const [authoritySaveState, setAuthoritySaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [authoritySaveMessage, setAuthoritySaveMessage] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const captureIntervalRef = useRef<number | null>(null)
  const isProcessingRef = useRef(false)
  const hasRequestedCameraPermission = useRef(false)

  useEffect(() => {
    return () => {
      stopLiveClassification()
    }
  }, [])

  useEffect(() => {
    setIntervalInputValue(String(classificationIntervalSeconds))
  }, [classificationIntervalSeconds])

  const fetchAuthorityContact = useCallback(async () => {
    setAuthorityLoading(true)
    setAuthorityLoadError(null)

    try {
      const response = await fetch(`${API_URL}/settings/authority-number`)
      let payload: { authority_number?: string | null; authority_email?: string | null; detail?: string } | null = null

      try {
        payload = await response.json()
      } catch {
        payload = null
      }

      if (!response.ok) {
        const detail = typeof payload?.detail === 'string' ? payload.detail : 'Failed to load authority contact.'
        throw new Error(detail)
      }

      const storedNumber = payload?.authority_number
      if (typeof storedNumber === 'string' && storedNumber.startsWith('+63') && storedNumber.length > 3) {
        setAuthorityNumber(storedNumber.slice(3))
        setAuthorityNumberError(null)
      } else {
        setAuthorityNumber('')
      }

      const storedEmail = payload?.authority_email
      if (typeof storedEmail === 'string' && storedEmail.length > 0) {
        setAuthorityEmail(storedEmail)
      } else {
        setAuthorityEmail('')
      }
      setAuthorityEmailError(null)

      setAuthoritySaveState('idle')
      setAuthoritySaveMessage(null)
      setAuthorityLoadError(null)

      if (!backendOnline) {
        setBackendOnline(true)
      }
    } catch (error) {
      console.error(error)
      setAuthorityLoadError(error instanceof Error ? error.message : 'Unable to load authority contact.')
      if (error instanceof TypeError) {
        setBackendOnline(false)
      }
    } finally {
      setAuthorityLoading(false)
    }
  }, [backendOnline])

  useEffect(() => {
    void fetchAuthorityContact()
  }, [fetchAuthorityContact])

  const fetchHistory = useCallback(async (filter: HistoryFilter) => {
    setHistoryLoading(true)
    setHistoryError(null)

    try {
      const params = new URLSearchParams({ limit: String(HISTORY_LIMIT) })
      if (filter !== 'all') {
        params.append('source', filter)
      }

      const response = await fetch(`${API_URL}/history?${params.toString()}`)
      let payload: { entries?: HistoryEntry[]; detail?: string } | null = null

      try {
        payload = await response.json()
      } catch {
        payload = null
      }

      if (!response.ok) {
        const detail = typeof payload?.detail === 'string' ? payload.detail : 'Failed to fetch history log.'
        throw new Error(detail)
      }

      if (!backendOnline) {
        setBackendOnline(true)
      }

      setHistoryEntries(payload?.entries ?? [])
    } catch (error) {
      console.error(error)
      setHistoryError(error instanceof Error ? error.message : 'Unable to load history log.')
      if (error instanceof TypeError) {
        setBackendOnline(false)
      }
    } finally {
      setHistoryLoading(false)
    }
  }, [backendOnline])

  useEffect(() => {
    if (!isHistoryOpen) return
    void fetchHistory(historyFilter)
  }, [isHistoryOpen, historyFilter, fetchHistory])

  const handleOpenHistory = () => {
    setIsHistoryOpen(true)
  }

  const handleCloseHistory = () => {
    setIsHistoryOpen(false)
  }

  const handleRefreshHistory = () => {
    void fetchHistory(historyFilter)
  }

  const getSourceLabel = (source?: string | null) => {
    if (source === 'camera') return 'Live Camera'
    if (source === 'upload') return 'Manual Upload'
    return source ?? 'Unknown Source'
  }

  const formatHistoryTimestamp = (entry: HistoryEntry) => {
    const timestamp = entry.logged_at ?? entry.created_at
    if (!timestamp) return 'Unknown time'
    return new Date(timestamp).toLocaleString()
  }

  const handleAuthorityNumberChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const digitsOnly = event.target.value.replace(/\D/g, '')
    const trimmed = digitsOnly.slice(0, PH_LOCAL_NUMBER_LENGTH)
    setAuthorityNumber(trimmed)
    setAuthoritySaveState('idle')
    setAuthoritySaveMessage(null)
    setAuthorityLoadError(null)

    if (!trimmed) {
      setAuthorityNumberError('Enter a contact number to notify.')
    } else if (trimmed.length < PH_LOCAL_NUMBER_LENGTH) {
      setAuthorityNumberError('Must include 10 digits after +63.')
    } else {
      setAuthorityNumberError(null)
    }
  }

  const handleAuthorityEmailChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value
    setAuthorityEmail(value)
    setAuthoritySaveState('idle')
    setAuthoritySaveMessage(null)
    setAuthorityLoadError(null)

    if (!value.trim()) {
      setAuthorityEmailError(null)
      return
    }

    if (!EMAIL_PATTERN.test(value.trim().toLowerCase())) {
      setAuthorityEmailError('Enter a valid email address or leave blank.')
    } else {
      setAuthorityEmailError(null)
    }
  }

  const handleSaveAuthorityContact = useCallback(async () => {
    if (authorityNumberError || authorityNumber.length < PH_LOCAL_NUMBER_LENGTH) {
      setAuthoritySaveState('error')
      setAuthoritySaveMessage('Provide a full Philippine number before saving.')
      return
    }

    if (authorityEmailError) {
      setAuthoritySaveState('error')
      setAuthoritySaveMessage('Fix the email field before saving.')
      return
    }

    const formattedAuthority = `+63${authorityNumber}`
    const trimmedEmail = authorityEmail.trim().toLowerCase()
    setAuthoritySaveState('saving')
    setAuthoritySaveMessage(null)

    try {
      const response = await fetch(`${API_URL}/settings/authority-number`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          authority_number: formattedAuthority,
          authority_email: trimmedEmail.length > 0 ? trimmedEmail : '',
        }),
      })

      let payload: { authority_number?: string; authority_email?: string | null; detail?: string } | null = null

      try {
        payload = await response.json()
      } catch {
        payload = null
      }

      if (!response.ok) {
        const detail = typeof payload?.detail === 'string' ? payload.detail : 'Failed to save authority contact.'
        throw new Error(detail)
      }

      const storedNumber = payload?.authority_number
      if (typeof storedNumber === 'string' && storedNumber.startsWith('+63') && storedNumber.length > 3) {
        setAuthorityNumber(storedNumber.slice(3))
      }

      const storedEmail = payload?.authority_email
      if (typeof storedEmail === 'string' && storedEmail.length > 0) {
        setAuthorityEmail(storedEmail)
      } else {
        setAuthorityEmail('')
      }

      setAuthoritySaveState('saved')
      setAuthoritySaveMessage('Contact details saved')

      if (!backendOnline) {
        setBackendOnline(true)
      }
    } catch (error) {
      console.error(error)
      setAuthoritySaveState('error')
      setAuthoritySaveMessage(error instanceof Error ? error.message : 'Unable to save authority contact.')
      if (error instanceof TypeError) {
        setBackendOnline(false)
      }
    }
  }, [authorityEmail, authorityEmailError, authorityNumber, authorityNumberError, backendOnline])

  const sendAuthorityEmailAlert = useCallback(async (payload: {
    recipient: string
    prediction: string
    status: string
    source: string
    filename?: string | null
  }) => {
    if (!EMAIL_ALERTS_ENABLED) return

    const templateParams = {
      to_email: payload.recipient,
      prediction: payload.prediction,
      status: payload.status,
      source: payload.source,
      filename: payload.filename ?? 'N/A',
      timestamp: new Date().toLocaleString(),
    }

    try {
      await emailjs.send(EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, templateParams, {
        publicKey: EMAILJS_PUBLIC_KEY,
      })
    } catch (error) {
      console.error('Authority email notification failed:', error)
    }
  }, [])


  const classifyBlob = useCallback(async (blob: Blob, origin: 'upload' | 'camera', filename?: string) => {
    if (isProcessingRef.current) return
    isProcessingRef.current = true
    setLoading(true)

    const formData = new FormData()
    formData.append('file', blob, 'frame.jpg')
    const formattedAuthority = authorityNumber.length === PH_LOCAL_NUMBER_LENGTH ? `+63${authorityNumber}` : ''
    if (formattedAuthority) {
      formData.append('authority_number', formattedAuthority)
    }
    const trimmedEmail = authorityEmail.trim().toLowerCase()
    const hasValidAuthorityEmail = Boolean(trimmedEmail && EMAIL_PATTERN.test(trimmedEmail))
    if (hasValidAuthorityEmail) {
      formData.append('authority_email', trimmedEmail)
    }

    try {
      const response = await fetch(`${API_URL}/predict?source=${encodeURIComponent(origin)}`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Failed to analyze image')
      if (!backendOnline) setBackendOnline(true)

      const data = await response.json()
      setPrediction(data.prediction)
      setStatus(data.status)
      setLastChecked(new Date())

      if (data.alert && data.prediction === 'Large Growth' && hasValidAuthorityEmail && EMAIL_ALERTS_ENABLED) {
        void sendAuthorityEmailAlert({
          recipient: trimmedEmail,
          prediction: data.prediction ?? 'Unknown Prediction',
          status: data.status ?? 'Warning',
          source: origin,
          filename: filename ?? (origin === 'upload' ? 'Uploaded image' : 'Live camera frame'),
        })
      }
    } catch (error) {
      console.error(error)
      alert('Error analyzing image')
      setBackendOnline(false)
    } finally {
      setLoading(false)
      isProcessingRef.current = false
    }
  }, [backendOnline, authorityEmail, authorityNumber, sendAuthorityEmailAlert])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (e) => {
      setImageSrc(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    await classifyBlob(file, 'upload', file.name)
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  const waitForVideoReady = (video: HTMLVideoElement) => {
    if (video.readyState >= 2) {
      return Promise.resolve()
    }

    return new Promise<void>((resolve) => {
      const handler = () => {
        video.removeEventListener('loadeddata', handler)
        resolve()
      }
      video.addEventListener('loadeddata', handler, { once: true })
    })
  }

  const startLiveClassification = async (options?: { force?: boolean }) => {
    if (isCameraRunning && !options?.force) return

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera API is not supported in this browser.')
      return
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: selectedCameraId ? { deviceId: { exact: selectedCameraId } } : { facingMode: 'environment' },
        audio: false,
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)

      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        await waitForVideoReady(videoRef.current)
      }

      setIsCameraRunning(true)
      setCameraError(null)

      await captureAndClassifyFrame()
    } catch (error) {
      console.error(error)
      setCameraError('Unable to access camera. Please grant permission and ensure a camera is connected.')
    }
  }

  const stopLiveClassification = () => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setIsCameraRunning(false)
  }

  const ensureCameraPermission = useCallback(async () => {
    if (hasRequestedCameraPermission.current) {
      return true
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      return false
    }

    try {
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true })
      tempStream.getTracks().forEach((track) => track.stop())
      hasRequestedCameraPermission.current = true
      return true
    } catch (error) {
      console.error(error)
      return false
    }
  }, [])

  const loadCameraDevices = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setDeviceLoadError('Camera enumeration is not supported in this browser.')
      setCameraDevices([])
      return
    }

    setDevicesLoading(true)
    setDeviceLoadError(null)

    const permissionGranted = await ensureCameraPermission()
    if (!permissionGranted) {
      setDevicesLoading(false)
      setDeviceLoadError('Camera permission is required to list sources.')
      setCameraDevices([])
      return
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const videoDevices: CameraDevice[] = devices
        .filter((device) => device.kind === 'videoinput')
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${index + 1}`,
        }))

      setCameraDevices(videoDevices)

      if (!videoDevices.length) {
        setSelectedCameraId(null)
        return
      }

      const matchingDevice = videoDevices.find((device) => device.deviceId === selectedCameraId)
      if (!matchingDevice) {
        setSelectedCameraId(videoDevices[0].deviceId)
      }
    } catch (error) {
      console.error(error)
      setDeviceLoadError('Unable to fetch camera list. Ensure permissions are granted.')
    } finally {
      setDevicesLoading(false)
    }
  }

  const handleOpenSettings = () => {
    setIsSettingsOpen(true)
    void loadCameraDevices()
  }

  const handleCloseSettings = () => {
    setIsSettingsOpen(false)
  }

  const handleSaveSettings = async () => {
    setIsSettingsOpen(false)

    if (isCameraRunning) {
      stopLiveClassification()
      await startLiveClassification({ force: true })
    }
  }

  const captureAndClassifyFrame = useCallback(async () => {
    if (!videoRef.current) return
    const video = videoRef.current

    if (!video.videoWidth || !video.videoHeight) return

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    const dataUrl = canvas.toDataURL('image/jpeg')
    setImageSrc(dataUrl)

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob((result) => resolve(result), 'image/jpeg', 0.85)
    )

    if (blob) {
      await classifyBlob(blob, 'camera', 'Live camera frame')
    }
  }, [classifyBlob])

  useEffect(() => {
    if (!isCameraRunning) {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current)
        captureIntervalRef.current = null
      }
      return
    }

    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
    }

    captureIntervalRef.current = window.setInterval(() => {
      captureAndClassifyFrame()
    }, classificationIntervalSeconds * 1000)

    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current)
        captureIntervalRef.current = null
      }
    }
  }, [isCameraRunning, classificationIntervalSeconds, captureAndClassifyFrame])

  const handleModeChange = (nextMode: 'upload' | 'camera') => {
    setMode(nextMode)
    if (nextMode === 'upload') {
      stopLiveClassification()
    }
  }

  const handleIntervalChange = useCallback((value: number) => {
    const clamped = clampInterval(value)
    setClassificationIntervalSeconds(clamped)
  }, [])

  const handleIntervalInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { value } = event.target
    setIntervalInputValue(value)

    if (value === '') return

    const numericValue = Number(value)
    if (!Number.isNaN(numericValue)) {
      handleIntervalChange(numericValue)
    }
  }

  const handleIntervalInputBlur = () => {
    setIntervalInputValue(String(classificationIntervalSeconds))
  }

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">River Hyacinth Monitor</h1>
            <p className="text-slate-500">Live CCTV Feed Analysis</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-slate-500">System Status:</span>
            <Badge
              variant="default"
              className={backendOnline ? 'bg-green-500 hover:bg-green-600' : 'bg-red-500 hover:bg-red-600'}
            >
              {backendOnline ? 'Online' : 'Offline'}
            </Badge>
          </div>
        </header>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {/* Main Feed Area */}
          <Card className="md:col-span-2 border-slate-200 shadow-sm bg-white">
            <CardHeader>
              <div className="flex flex-col gap-4">
                <CardTitle className="flex items-center gap-2">
                  <Camera className="h-5 w-5" />
                  Live Feed
                </CardTitle>
                <div className="flex flex-wrap gap-3">
                  <Button
                    variant={mode === 'upload' ? 'default' : 'outline'}
                    className="flex items-center gap-2"
                    onClick={() => handleModeChange('upload')}
                  >
                    <UploadCloud className="h-4 w-4" />
                    Upload Image
                  </Button>
                  <Button
                    variant={mode === 'camera' ? 'default' : 'outline'}
                    className="flex items-center gap-2"
                    onClick={() => handleModeChange('camera')}
                  >
                    <Video className="h-4 w-4" />
                    Live Camera
                  </Button>
                </div>
              </div>
              <CardDescription>Real-time monitoring of river section A-1</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-video bg-slate-900 rounded-lg overflow-hidden relative flex items-center justify-center">
                {mode === 'camera' ? (
                  <>
                    <video
                      ref={videoRef}
                      className="w-full h-full object-cover"
                      autoPlay
                      muted
                      playsInline
                    />
                    {!isCameraRunning && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-white/80 bg-slate-900">
                        <Video className="h-12 w-12" />
                        <p>Camera idle. Start live classification.</p>
                      </div>
                    )}
                  </>
                ) : imageSrc ? (
                  <img src={imageSrc} alt="Feed" className="w-full h-full object-cover" />
                ) : (
                  <div className="text-slate-500 flex flex-col items-center">
                    <Camera className="h-12 w-12 mb-2 opacity-50" />
                    <p>No signal / Upload an image to simulate feed</p>
                  </div>
                )}
                
                {/* Overlay */}
                <div className="absolute top-4 right-4 bg-black/50 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
                  LIVE
                </div>
              </div>
              {cameraError && mode === 'camera' && (
                <p className="mt-3 text-sm text-red-600">{cameraError}</p>
              )}
            </CardContent>
            <CardFooter className="justify-between">
              <div className="text-sm text-slate-500">
                Last update: {lastChecked.toLocaleTimeString()}
              </div>
              {mode === 'upload' ? (
                <div className="flex gap-2">
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    className="hidden" 
                    accept="image/*"
                    onChange={handleFileUpload}
                  />
                  <Button variant="outline" onClick={triggerFileInput} disabled={loading} className="border-slate-200 text-slate-700 hover:bg-slate-50">
                    {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : null}
                    Upload Image
                  </Button>
                </div>
              ) : (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    className={`border-slate-200 text-slate-700 hover:bg-slate-50 ${
                      isCameraRunning ? 'bg-slate-900 text-white hover:bg-slate-800' : ''
                    }`}
                    onClick={isCameraRunning ? stopLiveClassification : () => { void startLiveClassification() }}
                  >
                    {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : null}
                    {isCameraRunning ? 'Stop Live Feed' : 'Start Live Feed'}
                  </Button>
                </div>
              )}
            </CardFooter>
          </Card>

          {/* Status Panel */}
          <div className="space-y-6">
            <Card className="border-slate-200 shadow-sm bg-white">
              <CardHeader>
                <CardTitle>Current Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className={`flex items-center gap-3 p-4 rounded-lg border ${status === 'Warning' ? 'bg-red-50 border-red-100 text-red-700' : 'bg-green-50 border-green-100 text-green-700'}`}>
                  {status === 'Warning' ? (
                    <AlertTriangle className="h-8 w-8" />
                  ) : (
                    <CheckCircle className="h-8 w-8" />
                  )}
                  <div>
                    <div className="font-bold text-lg">{status}</div>
                    <div className="text-sm opacity-90">
                      {status === 'Warning' ? 'Action Required' : 'Clear for Ferries'}
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="text-sm font-medium text-slate-500">Growth Level</div>
                  <div className="text-2xl font-bold text-slate-900">{prediction}</div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ${
                        prediction === 'Large Growth' ? 'bg-red-500 w-full' :
                        prediction === 'Moderate Growth' ? 'bg-orange-500 w-2/3' :
                        prediction === 'Low Growth' ? 'bg-yellow-500 w-1/3' :
                        'bg-green-500 w-0'
                      }`}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-slate-200 shadow-sm bg-white">
              <CardHeader>
                <CardTitle>Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-4">
                  <div className="space-y-1">
                    <label htmlFor="authority-contact" className="text-sm font-semibold text-slate-700">
                      Authority Contact Number
                    </label>
                    <div className="flex rounded-md shadow-sm">
                      <span className="inline-flex items-center rounded-l-md border border-r-0 border-slate-200 bg-slate-50 px-3 text-sm text-slate-500">
                        +63
                      </span>
                      <input
                        id="authority-contact"
                        type="tel"
                        inputMode="numeric"
                        pattern="[0-9]*"
                        value={authorityNumber}
                        onChange={handleAuthorityNumberChange}
                        className="flex-1 rounded-r-md border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-300"
                        placeholder="9123456789"
                        disabled={authorityLoading}
                      />
                    </div>
                    <p className="text-xs text-slate-500">
                      Enter the 10 digits after +63 so we know who to notify when alerts trigger.
                    </p>
                    {authorityNumber && !authorityNumberError && (
                      <p className="text-xs font-medium text-emerald-600">Ready to notify +63{authorityNumber}</p>
                    )}
                    {authorityNumberError && <p className="text-xs text-red-600">{authorityNumberError}</p>}
                  </div>

                  <div className="space-y-1">
                    <label htmlFor="authority-email" className="text-sm font-semibold text-slate-700">
                      Authority Contact Email
                    </label>
                    <input
                      id="authority-email"
                      type="email"
                      value={authorityEmail}
                      onChange={handleAuthorityEmailChange}
                      className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-900 focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-300"
                      placeholder="authority@example.com"
                      disabled={authorityLoading}
                    />
                    <p className="text-xs text-slate-500">We'll also email alerts to this address when warnings fire.</p>
                    {authorityEmail && !authorityEmailError && (
                      <p className="text-xs font-medium text-emerald-600">Ready to notify {authorityEmail}</p>
                    )}
                    {authorityEmailError && <p className="text-xs text-red-600">{authorityEmailError}</p>}
                  </div>

                  {authorityLoading && (
                    <div className="flex items-center gap-2 text-xs text-slate-500">
                      <RefreshCw className="h-3 w-3 animate-spin" />
                      <span>Loading saved contact...</span>
                    </div>
                  )}
                  {authorityLoadError && (
                    <p className="text-xs text-red-600">{authorityLoadError}</p>
                  )}

                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="border-slate-200 text-slate-700 hover:bg-slate-50"
                      onClick={() => {
                        void handleSaveAuthorityContact()
                      }}
                      disabled={
                        authorityLoading ||
                        authorityNumberError !== null ||
                        authorityNumber.length < PH_LOCAL_NUMBER_LENGTH ||
                        authorityEmailError !== null ||
                        authoritySaveState === 'saving'
                      }
                    >
                      <span className="flex items-center gap-2">
                        {authoritySaveState === 'saving' && <RefreshCw className="h-3 w-3 animate-spin" />}
                        {authoritySaveState === 'saving' ? 'Saving...' : 'Save Contact'}
                      </span>
                    </Button>
                    {authoritySaveState === 'saved' && (
                      <p className="text-xs text-emerald-600">{authoritySaveMessage ?? 'Contact saved.'}</p>
                    )}
                    {authoritySaveState === 'error' && authoritySaveMessage && (
                      <p className="text-xs text-red-600">{authoritySaveMessage}</p>
                    )}
                  </div>
                </div>
                <Button
                  variant="outline"
                  className="w-full justify-start border-slate-200 text-slate-700 hover:bg-slate-50"
                  onClick={handleOpenHistory}
                >
                  <History className="h-4 w-4" />
                  View History Log
                </Button>
                <div className="space-y-1">
                  <Button
                    variant="outline"
                    className="w-full justify-start border-slate-200 text-slate-700 hover:bg-slate-50"
                    onClick={handleOpenSettings}
                  >
                    <Settings2 className="h-4 w-4" />
                    Camera Settings
                  </Button>
                  <p className="text-xs text-slate-500">
                    {formatIntervalLabel(classificationIntervalSeconds)}
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

      </div>

      {isSettingsOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={handleCloseSettings}
        >
          <div
            className="w-full max-w-xl rounded-2xl bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">Camera Settings</h2>
                <p className="text-sm text-slate-500">Choose the video source and analysis cadence.</p>
              </div>
              <Button variant="ghost" size="icon" onClick={handleCloseSettings} aria-label="Close camera settings">
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-6 px-6 py-4">
              <section className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                  <Camera className="h-4 w-4" />
                  <span>Camera Source</span>
                </div>
                <p className="text-xs text-slate-500">Select which connected camera to use for the live feed.</p>
                <div className="max-h-48 space-y-2 overflow-y-auto pr-1">
                  {devicesLoading ? (
                    <p className="text-sm text-slate-500">Scanning for cameras...</p>
                  ) : cameraDevices.length ? (
                    cameraDevices.map((device) => (
                      <label
                        key={device.deviceId}
                        className="flex items-center gap-3 rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-700"
                      >
                        <input
                          type="radio"
                          name="camera-source"
                          className="h-4 w-4 accent-slate-900"
                          checked={selectedCameraId === device.deviceId}
                          onChange={() => setSelectedCameraId(device.deviceId)}
                        />
                        <span>{device.label}</span>
                      </label>
                    ))
                  ) : (
                    <p className="text-sm text-slate-500">No cameras detected. Plug in a camera and grant permissions.</p>
                  )}
                </div>
                {deviceLoadError && <p className="text-sm text-red-600">{deviceLoadError}</p>}
                <Button
                  variant="outline"
                  size="sm"
                  className="border-slate-200 text-slate-700 hover:bg-slate-50"
                  onClick={() => {
                    void loadCameraDevices()
                  }}
                >
                  Refresh List
                </Button>
              </section>

              <section className="space-y-3 border-t border-slate-100 pt-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                  <SlidersHorizontal className="h-4 w-4" />
                  <span>Classification Frequency</span>
                </div>
                <p className="text-xs text-slate-500">Controls how often frames are analyzed while the live feed is running.</p>
                <input
                  type="range"
                  min={MIN_CLASSIFICATION_INTERVAL}
                  max={MAX_CLASSIFICATION_INTERVAL}
                  step={5}
                  value={classificationIntervalSeconds}
                  onChange={(event) => handleIntervalChange(Number(event.target.value))}
                  className="w-full accent-slate-900"
                />
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <label className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Manual entry (seconds)
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={MIN_CLASSIFICATION_INTERVAL}
                      max={MAX_CLASSIFICATION_INTERVAL}
                      value={intervalInputValue}
                      onChange={handleIntervalInputChange}
                      onBlur={handleIntervalInputBlur}
                      className="w-32 rounded-md border border-slate-200 px-3 py-1 text-sm text-slate-900 shadow-sm focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-300"
                    />
                    <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">sec</span>
                  </div>
                </div>
                <div className="flex justify-between text-xs font-medium uppercase tracking-wide text-slate-400">
                  <span>5s</span>
                  <span>24h</span>
                </div>
                <div className="text-sm font-medium text-slate-700">
                  {formatIntervalLabel(classificationIntervalSeconds)}
                </div>
                <p className="text-xs text-slate-500">
                  Changes apply immediately to the live feed interval. Click Apply to restart the video stream with the selected camera.
                </p>
              </section>
            </div>

            <div className="flex justify-end gap-2 border-t border-slate-100 bg-slate-50 px-6 py-4">
              <Button variant="ghost" onClick={handleCloseSettings}>
                Cancel
              </Button>
              <Button
                onClick={() => {
                  void handleSaveSettings()
                }}
              >
                Apply Settings
              </Button>
            </div>
          </div>
        </div>
      )}

      {isHistoryOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          onClick={handleCloseHistory}
        >
          <div
            className="w-full max-w-3xl rounded-2xl bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-slate-100 p-2 text-slate-700">
                  <History className="h-4 w-4" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Prediction History</h2>
                  <p className="text-sm text-slate-500">Review the latest logged classifications.</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {( ['all', 'camera', 'upload'] as HistoryFilter[]).map((filter) => (
                  <Button
                    key={filter}
                    variant={historyFilter === filter ? 'default' : 'outline'}
                    size="sm"
                    className={historyFilter === filter ? 'bg-slate-900 text-white hover:bg-slate-800' : 'border-slate-200 text-slate-700'}
                    onClick={() => setHistoryFilter(filter)}
                  >
                    {HISTORY_FILTER_LABELS[filter]}
                  </Button>
                ))}
                <Button
                  variant="outline"
                  size="sm"
                  className="border-slate-200 text-slate-700"
                  onClick={handleRefreshHistory}
                  disabled={historyLoading}
                >
                  <RefreshCw className={`h-4 w-4 ${historyLoading ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
                <Button variant="ghost" size="icon" onClick={handleCloseHistory} aria-label="Close history log">
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div className="max-h-[65vh] space-y-4 overflow-y-auto px-6 py-4">
              {historyLoading ? (
                <div className="flex h-40 flex-col items-center justify-center gap-2 text-slate-500">
                  <RefreshCw className="h-5 w-5 animate-spin" />
                  <p>Loading recent activity...</p>
                </div>
              ) : historyError ? (
                <div className="rounded-lg border border-red-100 bg-red-50 p-4 text-sm text-red-700">
                  <p>{historyError}</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-3 border-red-200 text-red-700 hover:bg-red-100"
                    onClick={handleRefreshHistory}
                  >
                    Try Again
                  </Button>
                </div>
              ) : historyEntries.length === 0 ? (
                <div className="flex h-40 flex-col items-center justify-center gap-2 text-slate-500">
                  <History className="h-5 w-5" />
                  <p>No predictions have been logged yet.</p>
                </div>
              ) : (
                <ul className="divide-y divide-slate-100">
                  {historyEntries.map((entry, index) => {
                    const key = String(entry.id ?? entry.logged_at ?? `${entry.filename ?? 'entry'}-${index}`)
                    const isWarning = entry.status === 'Warning'
                    return (
                      <li key={key} className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between">
                        <div className="space-y-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <p className="text-sm font-semibold text-slate-900">{entry.prediction ?? 'Unknown Prediction'}</p>
                            <Badge variant="outline" className="text-xs capitalize">
                              {getSourceLabel(entry.source)}
                            </Badge>
                          </div>
                          <p className="text-xs text-slate-500">
                            {formatHistoryTimestamp(entry)}
                            {entry.filename ? ` • ${entry.filename}` : ''}
                            {entry.authority_number ? ` • Notified: ${entry.authority_number}` : ''}
                            {entry.authority_email ? ` • Email: ${entry.authority_email}` : ''}
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-sm font-semibold">
                          {isWarning ? (
                            <AlertTriangle className="h-4 w-4 text-red-600" />
                          ) : (
                            <CheckCircle className="h-4 w-4 text-emerald-600" />
                          )}
                          <span className={isWarning ? 'text-red-600' : 'text-emerald-600'}>
                            {entry.status ?? 'Unknown'}
                          </span>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
