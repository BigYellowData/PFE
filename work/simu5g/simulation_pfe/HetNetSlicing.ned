// =============================================================================
// HetNetSlicing.ned
// Réseau 5G Hétérogène — 3 gNodeBs + Architecture MEC distribuée
//
// Conformité normative :
//   3GPP TR 38.901 Table 7.2-1  — Urban Macro (UMa) et Urban Micro (UMi)
//                                  deployment models
//   3GPP TS 22.261 §6.22        — Private 5G networks (verticals industriels)
//   3GPP TS 38.423              — Xn Application Protocol (inter-gNB)
//   ETSI GS MEC 003             — Multi-access Edge Computing reference arch.
//   3GPP TS 23.501 §5.15        — Network Slicing (eMBB/URLLC/mMTC)
//
// Topologie :
//   gNB_Macro    (UMa)             — couverture large, 500m ISD, mix équilibré
//   gNB_Commerce (UMi StreetCanyon) — zone dense 150m, eMBB dominant
//   gNB_Industrie (UMi Indoor)      — campus privé 200m, URLLC+mMTC dominant
//
// Core réseau :
//   iUpf_MEC — edge UPF publique, dessert Macro + Commerce (MEC partagé)
//   iUpf_Ind — edge UPF industrielle privée, dessert Industrie seule
//   upf      — anchor UPF partagé (N9 → N6)
//   router   — routeur Data Network (N6 → serveurs applicatifs)
//
// Interfaces radio :
//   Xn full-mesh entre les 3 gNodeBs (handover + load balancing)
// =============================================================================



import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.world.radio.LteChannelControl;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.common.binder.Binder;
import simu5g.nodes.gNodeB;
import simu5g.nodes.NrUe;
import simu5g.nodes.Upf;
import simu5g.control.SliceController;

network HetNetSlicing
{
    parameters:
        // ── Cellule A : gNB_Macro (Urban Macro — 3GPP UMa) ───────────────────
        // Couverture large, mix équilibré entre les trois slices
        int numEmbbUes_A  = default(4);   // eMBB : cloud gaming, streaming HD
        int numUrllcUes_A = default(4);   // URLLC : VoLTE, contrôle temps-réel
        int numMmtcUes_A  = default(2);   // mMTC : IoT résidentiel, compteurs

        // ── Cellule B : gNB_Commerce (Urban Micro StreetCanyon) ───────────────
        // Zone commerciale dense : eMBB largement dominant
        int numEmbbUes_B  = default(6);   // eMBB dominant : réseaux sociaux, streaming
        int numUrllcUes_B = default(2);   // URLLC : paiements NFC, contrôle accès
        int numMmtcUes_B  = default(1);   // mMTC : capteurs vitrine, affiches connectées

        // ── Cellule C : gNB_Industrie (UMi Indoor — réseau privé 5G) ──────────
        // Campus industriel : URLLC et mMTC largement dominants
        // Réf: 3GPP TS 22.261 §6.22 — Verticals (Industry 4.0)
        int numEmbbUes_C  = default(2);   // eMBB : supervision vidéo HD
        int numUrllcUes_C = default(4);   // URLLC dominant : automates, AGV, cobots
        int numMmtcUes_C  = default(6);   // mMTC dominant : capteurs IoT, SCADA

        @display("bgb=1700,1050");

    submodules:

        // ── Infrastructure simu5G (modules systèmes) ─────────────────────────
        channelControl: LteChannelControl {
            @display("p=80,40;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=260,40;is=s");
        }
        binder: Binder {
            @display("p=440,40;is=s");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=620,40;is=s");
        }

        // ── Closed-loop slice controller (PFE) ───────────────────────────────
        sliceController: SliceController {
            controlPeriod = 1s;       // 1s period = write metrics every second
            rbConfigFile  = "rb_config.json";
            metricsFile   = "metrics_live.csv";
            @display("p=800,40;is=s");
        }

        // ── gNodeBs ───────────────────────────────────────────────────────────

        // Cellule A — Macro (Urban Macro, 3GPP TR 38.901 Table 7.2-1)
        // Paramètres radio : 3.5 GHz n78, 40 dBm, h_BS=25m, ISD=500m
        gnbMacro: gNodeB {
            @display("p=530,450;is=vl;i=device/antennatower");
        }

        // Cellule B — Commerce (Urban Micro StreetCanyon, 3GPP TR 38.901)
        // Paramètres radio : 3.5 GHz n78, 30 dBm, h_BS=10m, ISD=150m
        gnbCommerce: gNodeB {
            @display("p=230,240;is=l;i=device/antenna");
        }

        // Cellule C — Industrie (Urban Micro Indoor, réseau privé 5G)
        // Paramètres radio : 3.5 GHz n78, 33 dBm, h_BS=6m (intérieur)
        // Réf: 3GPP TS 22.261 §6.22 — Industry 4.0 private network
        gnbIndustrie: gNodeB {
            @display("p=840,680;is=l;i=device/antenna");
        }

        // ── Core réseau 5G — Architecture MEC distribuée ─────────────────────

        // Edge UPF publique — dessert gNB_Macro + gNB_Commerce
        // Réf: ETSI GS MEC 003 (MEC collocated with access node)
        iUpf_MEC: Upf {
            @display("p=800,340;is=l");
        }

        // Edge UPF industrielle privée — dessert gNB_Industrie seule
        // Réf: 3GPP TS 23.501 §5.33 (Local Area Data Network — LADN)
        iUpf_Ind: Upf {
            @display("p=1020,680;is=l");
        }

        // Anchor UPF partagé (interface N9 → N6)
        upf: Upf {
            @display("p=1200,480;is=l");
        }

        // Routeur N6 — Data Network
        router: Router {
            @display("p=1420,480;is=l");
        }

        // ── Serveurs applicatifs (un par type de slice) ───────────────────────
        serverEmbb: StandardHost {
            @display("p=1620,280;is=n");
        }
        serverUrllc: StandardHost {
            @display("p=1620,480;is=n");
        }
        serverMmtc: StandardHost {
            @display("p=1620,680;is=n");
        }

        // ── UEs Cellule A (gNB_Macro) ─────────────────────────────────────────
        // Zone géographique : rayon 400m autour de gnbMacro
        ueEmbb_A[numEmbbUes_A]: NrUe {
            @display("p=310,380");
        }
        ueUrllc_A[numUrllcUes_A]: NrUe {
            @display("p=310,460");
        }
        ueMmtc_A[numMmtcUes_A]: NrUe {
            @display("p=310,540");
        }

        // ── UEs Cellule B (gNB_Commerce) ──────────────────────────────────────
        // Zone géographique : rayon 150m autour de gnbCommerce
        ueEmbb_B[numEmbbUes_B]: NrUe {
            @display("p=80,160");
        }
        ueUrllc_B[numUrllcUes_B]: NrUe {
            @display("p=80,240");
        }
        ueMmtc_B[numMmtcUes_B]: NrUe {
            @display("p=80,320");
        }

        // ── UEs Cellule C (gNB_Industrie) ─────────────────────────────────────
        // Zone géographique : rayon 200m autour de gnbIndustrie (campus)
        ueEmbb_C[numEmbbUes_C]: NrUe {
            @display("p=660,750");
        }
        ueUrllc_C[numUrllcUes_C]: NrUe {
            @display("p=660,830");
        }
        ueMmtc_C[numMmtcUes_C]: NrUe {
            @display("p=660,910");
        }

    connections allowunconnected:

        // ── Interfaces Xn — Coordination inter-cellules (3GPP TS 38.423) ─────
        // Full mesh : chaque paire de gNodeBs est directement connectée
        // Utilisé pour : handover, load balancing, CoMP, coordination RRM
        gnbMacro.x2++    <--> Eth10G <--> gnbCommerce.x2++;
        gnbMacro.x2++    <--> Eth10G <--> gnbIndustrie.x2++;
        gnbCommerce.x2++ <--> Eth10G <--> gnbIndustrie.x2++;

        // ── Interfaces N3 — gNodeB → edge UPF (3GPP TS 38.415 / GTP-U) ──────
        // Macro + Commerce → MEC UPF publique
        gnbMacro.ppp    <--> Eth10G <--> iUpf_MEC.pppg++;
        gnbCommerce.ppp <--> Eth10G <--> iUpf_MEC.pppg++;
        // Industrie → MEC UPF privée (trafic industriel isolé)
        gnbIndustrie.ppp <--> Eth10G <--> iUpf_Ind.pppg++;

        // ── Interfaces N9 — edge UPF → anchor UPF ─────────────────────────────
        iUpf_MEC.pppg++ <--> Eth10G <--> upf.pppg++;
        iUpf_Ind.pppg++ <--> Eth10G <--> upf.pppg++;

        // ── Interface N6 — anchor UPF → Data Network ─────────────────────────
        upf.filterGate <--> Eth10G <--> router.pppg++;

        // ── Data Network — Router → Serveurs applicatifs ──────────────────────
        router.pppg++ <--> Eth10G <--> serverEmbb.pppg++;
        router.pppg++ <--> Eth10G <--> serverUrllc.pppg++;
        router.pppg++ <--> Eth10G <--> serverMmtc.pppg++;
}
