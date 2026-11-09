package userstate

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Keys of the nearline generative recall (design §11 step 5, §12.2), all in the state instance.
// ml/nearline/keys.py is the Python side; testdata/gen_payload.json pins the payload for both.
//
//	gen:worker          a generation worker is alive (it refreshes the key, PX 30 s); apply.lua
//	                    queues history changes only while it exists
//	gen:queue           ZSET uid -> due (ms)
//	gen:pending:{uid}   LIST "<event ms>:<applied ms>": the changes not picked up yet
//	gen:{uid}:{ver}     STRING, JSON Generated: the worker's result for the user
const (
	GenWorkerKey = "gen:worker"
	GenQueueKey  = "gen:queue"
)

func GenPendingKey(uid int64) string { return fmt.Sprintf("gen:pending:%d", uid) }

// GenKey is the result of generator version ver (e.g. minionerec-365a03fc-b50) for a user.
func GenKey(uid int64, ver string) string { return fmt.Sprintf("gen:%d:%s", uid, ver) }

// Generated is one result of the generation worker: the top items generated from the user's
// history at version HV, best first, with their beam scores (log-probabilities).
type Generated struct {
	HV     int64     `json:"hv"`
	At     int64     `json:"at"` // when it was generated (ms since the epoch, Redis clock)
	Items  []int32   `json:"items"`
	Scores []float64 `json:"scores"`
}

// AtTime is At as a time.
func (g Generated) AtTime() time.Time { return time.UnixMilli(g.At) }

// DecodeGenerated parses a gen:{uid}:{ver} value.
func DecodeGenerated(raw []byte) (Generated, error) {
	var g Generated
	if err := json.Unmarshal(raw, &g); err != nil {
		return Generated{}, err
	}
	if len(g.Scores) != len(g.Items) {
		return Generated{}, errors.New("gen payload: items and scores differ in length")
	}
	return g, nil
}
